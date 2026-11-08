"""My MuJoCo simulation of the cell for drawing AMZ-SA-26-001.

    python simulation.py              starts the 3D view (and prints the numbers)
    python simulation.py --no-view    only the numbers

Two xArm6 robots put the four profiles into my fixture. Robot 1 (blue flange) places Pos 1a
and Pos 2a, robot 2 (pink) places Pos 1b and Pos 2b. Only one robot may work at the fixture
at a time. I use the simulation to check the motions and the order of the steps. It does not
show the real accuracy; for that I would test the fixture with real profiles.

Units are m, rad and s. The origin is the middle of the fixture plate top, x runs along Pos 1,
y along Pos 2, z up. In the code I call robot 1 "A" and robot 2 "B".
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MESHES = Path(__file__).resolve().parent / "xarm6_meshes"

# --- parts ----------------------------------------------------------------------------------------
PROFILE = {"Pos1": (0.320, 0.040, 0.020), "Pos2": (0.280, 0.030, 0.015)}   # length, width, height
WALL = 0.002
FIXTURE_POSE = {                      # where each part goes: x, y, z of the leg ends, turn about z
    "Pos1_a": (0.0, -0.060, 0.0, 0.0),
    "Pos1_b": (0.0, 0.060, 0.0, 0.0),
    "Pos2_a": (-0.060, 0.0, 0.020, np.pi / 2),
    "Pos2_b": (0.060, 0.0, 0.020, np.pi / 2),
}
JOBS = {"A": ["Pos1_a", "Pos2_a"], "B": ["Pos1_b", "Pos2_b"]}
NAME = {"A": "robot 1", "B": "robot 2"}
# I put the parts a bit off in the rack, so the jaws must push them
OFFSET = {"Pos1_a": 0.0015, "Pos1_b": -0.0010, "Pos2_a": -0.0015, "Pos2_b": 0.0010}
JAW_OPEN = 0.003                      # I open the jaws 3 mm, so a part may land up to 2 mm off

# --- layout ---------------------------------------------------------------------------------------
BASE_DIST = 0.48                      # I put each robot base 480 mm from the fixture centre
TABLE = -0.020                        # table top, the 20 mm plate sits on it
RACK = {"Pos1": (0.28, -0.41), "Pos2": (0.36, -0.41)}   # rack of robot 1, I mirror it for robot 2
RACK_Z = 0.020                        # parts in the rack lie on 20 mm blocks

# --- xArm6 (joint frames from UFACTORY xarm_ros2) and gripper -------------------------------------
JOINT_XYZ = [(0, 0, 0.267), (0, 0, 0), (0.0535, -0.2845, 0), (0.0775, 0.3425, 0), (0, 0, 0), (0.076, 0.097, 0)]
JOINT_ROLL = [0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2]
Q_MIN = np.radians([-360, -117, -219, -360, -97, -360])
Q_MAX = np.radians([360, 116, 10, 360, 180, 360])
TCP = 0.172                           # tool point on the flange axis
FINGER_MAX = 0.042                    # half the 84 mm stroke of the gripper

# --- motion ---------------------------------------------------------------------------------------
QD_MAX, QDD_MAX = 2.0, 6.0            # joint speed rad/s, acceleration rad/s^2
SLOW = 0.25                           # I go at quarter speed for the last 60 mm to a part
APPROACH = 0.060
GRASP_DEPTH = 0.008                   # fingers 8 mm below the top of the profile
# my guesses for these times, I would measure them on the real cell
T_GRIP, T_OPEN, T_JAWS = 0.5, 0.4, 0.3  # s: gripper closes, gripper opens, one set of jaws closes
DT = 0.05                             # 20 frames per second

COLOR = {"plate": "0.74 0.76 0.78 1", "jaw": "0.24 0.30 0.37 1", "endjaw": "0.93 0.63 0.00 1",
         "gripper": "0.60 0.63 0.66 1", "drive": "0.36 0.47 0.60 1", "pinion": "0.36 0.47 0.60 1",
         "rack": "0.52 0.55 0.58 1", "table": "0.86 0.85 0.82 1",
         "robot": "0.94 0.94 0.93 1", "hand": "0.20 0.20 0.22 1", "finger": "0.38 0.38 0.41 1",
         "A": "0.165 0.471 0.839 1", "B": "0.910 0.482 0.643 1",
         "Pos1": "0.922 0.408 0.204 1", "Pos2": "0.106 0.686 0.478 1"}


# --- scene ----------------------------------------------------------------------------------------
def num(*values):
    return " ".join(f"{v:.6g}" for v in values)


def box(pos, half, color, name=""):
    name = f'name="{name}" ' if name else ""
    if color == "pinion":                                               # a cylinder
        return f'<geom {name}type="cylinder" pos="{num(*pos)}" size="{num(half[0], half[2])}" rgba="{COLOR[color]}"/>'
    return f'<geom {name}type="box" pos="{num(*pos)}" size="{num(*half)}" rgba="{COLOR[color]}"/>'


def fixture_boxes():
    """My fixture as simple boxes (centre, half size, kind, motion), jaws closed. I build the
    CAD file from the same list. motion is None for fixed parts, (part, "side" or "end",
    outward direction) for a jaw and everything that moves with it, and (part, "pinion", None)
    for a pinion. Side jaws: each pair is a pneumatic parallel gripper. End jaws: each sits on
    a slide under the plate, the two racks of a profile turn one pinion, and one air cylinder
    moves them."""
    b = [((0, 0, -0.010), (0.210, 0.190, 0.010), "plate", None)]        # plate 420 x 380 x 20 mm
    for part, y in (("Pos1_a", -0.060), ("Pos1_b", 0.060)):            # Pos 1 along x, on the plate
        for x in (-0.125, 0.125):
            b.append(((x, y, -0.040), (0.020, 0.045, 0.020), "gripper", None))           # under the plate
            for s in (-1, 1):
                move = (part, "side", (0, s))
                b.append(((x, y + s * 0.026, 0.009), (0.010, 0.006, 0.007), "jaw", move))
                b.append(((x, y + s * 0.029, -0.011), (0.006, 0.003, 0.013), "drive", move))   # finger
        for s in (-1, 1):                                               # end jaws, 44 mm wide
            move = (part, "end", (s, 0))
            b.append(((s * 0.165, y, 0.009), (0.005, 0.022, 0.007), "endjaw", move))
            b.append(((s * 0.166, y, -0.031), (0.004, 0.006, 0.033), "drive", move))     # down to the slide
            b.append(((s * 0.075, y - s * 0.008, -0.067), (0.095, 0.003, 0.004), "drive", move))  # rack
        b.append(((0.030, y + 0.008, -0.067), (0.010, 0.002, 0.002), "drive", (part, "end", (-1, 0))))  # rod
        b.append(((0.065, y + 0.008, -0.067), (0.035, 0.008, 0.008), "drive", None))     # air cylinder
        b.append(((0, y, -0.067), (0.005, 0.005, 0.004), "pinion", (part, "pinion", None)))
        b.append(((0.0045, y, -0.067), (0.0015, 0.001, 0.004), "drive", (part, "pinion", None)))  # tooth mark
    for part, x in (("Pos2_a", -0.060), ("Pos2_b", 0.060)):            # Pos 2 along y, on the Pos 1 webs
        for y in (-0.110, 0.110):
            b.append(((x, y, 0.009), (0.031, 0.010, 0.009), "gripper", None))            # on the plate
            for s in (-1, 1):
                b.append(((x + s * 0.021, y, 0.0255), (0.006, 0.010, 0.0075), "jaw", (part, "side", (s, 0))))
        for s in (-1, 1):                                               # end jaws, 34 mm wide
            move = (part, "end", (0, s))
            b.append(((x, s * 0.145, 0.016), (0.017, 0.005, 0.016), "endjaw", move))
            b.append(((x, s * 0.146, -0.015), (0.006, 0.004, 0.015), "drive", move))     # down to the slide
            b.append(((x - s * 0.008, s * 0.065, -0.029), (0.003, 0.085, 0.004), "drive", move))  # rack
        b.append(((x + 0.008, 0.030, -0.029), (0.002, 0.010, 0.002), "drive", (part, "end", (0, -1))))  # rod
        b.append(((x + 0.008, 0.065, -0.029), (0.008, 0.035, 0.008), "drive", None))     # air cylinder
        b.append(((x, 0, -0.029), (0.005, 0.005, 0.004), "pinion", (part, "pinion", None)))
        b.append(((x, 0.0045, -0.029), (0.001, 0.0015, 0.004), "drive", (part, "pinion", None)))  # tooth mark
    return b


def fixture_xml():
    """Fixed parts are plain geoms. The jaws on one side of a profile are one body on a slide
    (0 = closed, JAW_OPEN = open). Each pinion turns on a hinge."""
    fixed, moving = [], {}
    for i, (pos, half, kind, move) in enumerate(fixture_boxes()):
        g = box(pos, half, kind, f"fx{i}_{kind}")
        if move is None:
            fixed.append(g)
        else:
            moving.setdefault(move, []).append((pos, g))
    out = fixed
    for (part, group, d), geoms in moving.items():
        name = f"fx_{part}_{group}" + (f"_{d[0]}{d[1]}" if d else "")
        if group == "pinion":
            joint = f'<joint name="{name}" type="hinge" pos="{num(*geoms[0][0])}" axis="0 0 1"/>'
        else:
            joint = f'<joint name="{name}" type="slide" axis="{d[0]} {d[1]} 0" range="0 {JAW_OPEN}"/>'
        out += [f'<body name="{name}">', '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>',
                joint] + [g for _, g in geoms] + ["</body>"]
    return "\n".join(out)


def table_xml():
    """40 mm table top with a hole under the plate for the jaw drives."""
    X, Y, hx, hy, z = 0.75, 0.95, 0.185, 0.165, TABLE - 0.020
    pieces = [((-(X + hx) / 2, 0), ((X - hx) / 2, Y)), (((X + hx) / 2, 0), ((X - hx) / 2, Y)),
              ((0, -(Y + hy) / 2), (hx, (Y - hy) / 2)), ((0, (Y + hy) / 2), (hx, (Y - hy) / 2))]
    return "\n".join(box((x, y, z), (a, b, 0.020), "table") for (x, y), (a, b) in pieces)


def rack_xml(robot):
    s = 1 if robot == "A" else -1
    y = s * RACK["Pos1"][1]
    g = [box((s * 0.32, y, -0.010), (0.075, 0.175, 0.010), "rack")]
    for part, (x, _) in RACK.items():
        w = PROFILE[part][1]
        for dy in (-0.110, 0.110):
            g.append(box((s * x, y + dy, 0.010), (0.026, 0.010, 0.010), "rack"))              # block
            for side in (-1, 1):                                                              # guides
                g.append(box((s * x + side * (w / 2 + 0.005), y + dy, 0.026), (0.003, 0.010, 0.006), "rack"))
    return "\n".join(g)


def robot_xml(r):
    y, yaw = (-BASE_DIST, np.pi / 2) if r == "A" else (BASE_DIST, -np.pi / 2)
    out = [f'<body name="{r}_base" pos="0 {y} {TABLE}" quat="{num(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))}">',
           f'<geom name="{r}_link0" type="mesh" mesh="link_base" rgba="{COLOR["robot"]}"/>']
    for i in range(6):
        roll = JOINT_ROLL[i]
        out += [f'<body name="{r}_link{i + 1}" pos="{num(*JOINT_XYZ[i])}" '
                f'quat="{num(np.cos(roll / 2), np.sin(roll / 2), 0, 0)}">',
                '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>',
                f'<joint name="{r}_j{i + 1}" axis="0 0 1" range="{Q_MIN[i]:.5f} {Q_MAX[i]:.5f}"/>',
                f'<geom name="{r}_link{i + 1}" type="mesh" mesh="link{i + 1}" rgba="{COLOR["robot"]}"/>']
    # gripper: blue or pink ring, body, two fingers, tool point
    out += [f'<geom name="{r}_adapter" type="cylinder" pos="0 0 0.006" size="0.037 0.006" rgba="{COLOR[r]}"/>',
            f'<geom name="{r}_gripper" type="box" pos="0 0 0.066" size="0.030 0.046 0.054" '
            f'rgba="{COLOR["hand"]}"/>',
            f'<site name="{r}_tcp" pos="0 0 {TCP}" size="0.004" group="4"/>']
    for side, sign in (("l", 1), ("r", -1)):
        out += [f'<body name="{r}_finger_{side}">',
                '<inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>',
                f'<joint name="{r}_finger_{side}" type="slide" axis="0 {sign} 0" range="0 {FINGER_MAX}"/>',
                f'<geom name="{r}_finger_{side}" type="box" pos="0 {sign * 0.005} 0.150" '
                f'size="0.010 0.005 0.030" rgba="{COLOR["finger"]}"/>',
                '</body>']
    out += ["</body>"] * 7                                              # link6 ... link1, base
    return "\n".join(out)


def part_xml(name):
    kind = name[:4]
    L, w, h = PROFILE[kind]
    t = WALL
    return "\n".join([
        f'<body name="{name}" mocap="true">',                           # U-profile, legs down
        box((0, 0, h - t / 2), (L / 2, w / 2, t / 2), kind, f"{name}_web"),
        box((0, -(w - t) / 2, (h - t) / 2), (L / 2, t / 2, (h - t) / 2), kind, f"{name}_leg1"),
        box((0, (w - t) / 2, (h - t) / 2), (L / 2, t / 2, (h - t) / 2), kind, f"{name}_leg2"),
        "</body>"])


def scene_xml():
    meshes = "\n".join(f'<mesh name="{n}" file="{n}.stl"/>' for n in ["link_base"] + [f"link{i}" for i in range(1, 7)])
    legs = "\n".join(box((sx * 0.70, sy * 0.90, (TABLE - 0.84) / 2), (0.03, 0.03, (TABLE + 0.76) / 2), "rack")
                     for sx in (-1, 1) for sy in (-1, 1))
    return f"""<mujoco model="xarm6_cell">
  <compiler angle="radian" meshdir="{MESHES.as_posix()}" autolimits="true" inertiafromgeom="false"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="35"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.50 0.50 0.50" specular="0.1 0.1 0.1"/>
    <rgba haze="0.95 0.95 0.94 1"/>
  </visual>
  <asset>
    {meshes}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.99 0.99 0.98" rgb2="0.86 0.87 0.88"
             width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.95 0.95 0.93" rgb2="0.90 0.90 0.88"
             width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
  </asset>
  <default><geom contype="1" conaffinity="0"/></default>
  <worldbody>
    <light pos="1.2 -1.5 2.5" dir="-0.4 0.5 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <light pos="-1.5 1.0 2.0" dir="0.5 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom type="plane" pos="0 0 -0.80" size="4 4 0.1" material="floor"/>
    {table_xml()}
    {legs}
    {fixture_xml()}
    {rack_xml("A")}
    {rack_xml("B")}
    {robot_xml("A")}
    {robot_xml("B")}
    {chr(10).join(part_xml(p) for p in FIXTURE_POSE)}
  </worldbody>
</mujoco>"""


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def part_pose(x, y, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return pose([[c, -s, 0], [s, c, 0], [0, 0, 1]], (x, y, z))


def tool_down(x, y, z, yaw):
    """Tool point at (x, y, z), pointing down, fingers closing across a part turned by yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return pose([[c, s, 0], [s, -c, 0], [0, 0, -1]], (x, y, z))


def lifted(T, dz=APPROACH):
    T = T.copy()
    T[2, 3] += dz
    return T


def along(a):
    """Shift by a along the part."""
    return pose(np.eye(3), (a, 0, 0))


class Cell:
    """The MuJoCo model, plus where to find the robot joints, fingers, jaws and parts."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joints = {r: [m.joint(f"{r}_j{i}").qposadr[0] for i in range(1, 7)] for r in "AB"}
        self.dofs = {r: [m.joint(f"{r}_j{i}").dofadr[0] for i in range(1, 7)] for r in "AB"}
        self.fingers = {r: [m.joint(f"{r}_finger_{s}").qposadr[0] for s in "lr"] for r in "AB"}
        self.tcp_site = {r: m.site(f"{r}_tcp").id for r in "AB"}
        self.mocap = {p: m.body(p).mocapid[0] for p in FIXTURE_POSE}
        self.jaws, self.pinion = {}, {}      # (part, "side" or "end") -> joints, part -> (joint, radius)
        for j in range(m.njnt):
            bits = m.joint(j).name.split("_")
            if bits[0] != "fx":
                continue
            part, group, adr = f"{bits[1]}_{bits[2]}", bits[3], m.joint(j).qposadr[0]
            if group == "pinion":
                g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.jnt_bodyid[j]
                         and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
                self.pinion[part] = (adr, m.geom_size[g][0])
            else:
                self.jaws.setdefault((part, group), []).append(adr)

    def set_robot(self, r, q, opening):
        self.data.qpos[self.joints[r]] = q
        self.data.qpos[self.fingers[r]] = np.clip(opening / 2, 0.0, FINGER_MAX)

    def set_jaws(self, part, group, closed):
        """0 = open, 1 = closed. The end jaws also turn their pinion."""
        self.data.qpos[self.jaws[(part, group)]] = JAW_OPEN * (1 - closed)
        if group == "end":
            adr, radius = self.pinion[part]
            self.data.qpos[adr] = JAW_OPEN * closed / radius

    def set_part(self, name, T):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        self.data.mocap_pos[self.mocap[name]] = T[:3, 3]
        self.data.mocap_quat[self.mocap[name]] = quat

    def tcp(self, r):
        """Tool point of robot r. Run mujoco.mj_kinematics first."""
        s = self.tcp_site[r]
        return pose(self.data.site_xmat[s].reshape(3, 3), self.data.site_xpos[s])

    def fk(self, r, q):
        self.data.qpos[self.joints[r]] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.tcp(r)

    def show(self, frame):
        """Puts the cell into one saved frame."""
        self.data.qpos[:] = frame.qpos
        self.data.mocap_pos[:] = frame.mocap_pos
        self.data.mocap_quat[:] = frame.mocap_quat
        mujoco.mj_kinematics(self.model, self.data)


# --- inverse kinematics ---------------------------------------------------------------------------
def ik(cell, r, T_goal, q_start):
    """Damped least squares. Joint angles for the tool pose T_goal, or None."""
    m, d, site = cell.model, cell.data, cell.tcp_site[r]
    q = np.clip(np.array(q_start, float), Q_MIN, Q_MAX)
    jac_p, jac_r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(300):
        d.qpos[cell.joints[r]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.site_xmat[site].reshape(3, 3)
        err_pos = T_goal[:3, 3] - d.site_xpos[site]
        err_rot = 0.5 * sum(np.cross(R[:, i], T_goal[:3, i]) for i in range(3))
        if np.linalg.norm(err_pos) < 1e-5 and np.linalg.norm(err_rot) < 1e-4:
            return q
        mujoco.mj_jacSite(m, d, jac_p, jac_r, site)
        J = np.vstack([jac_p, jac_r])[:, cell.dofs[r]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.concatenate([err_pos, err_rot]))
        q = np.clip(q + np.clip(dq, -0.3, 0.3), Q_MIN, Q_MAX)
    return None


def solve(cell, r, T_goal, seeds, turn_ok=False):
    """I try a few start poses (and the tool turned by 180 deg if turn_ok, because the gripper
    is symmetric) and keep the answer closest to the first seed."""
    base = np.array([0.0, -BASE_DIST if r == "A" else BASE_DIST])
    v = T_goal[:2, 3] - base
    j1 = np.arctan2(-v[0], v[1]) if r == "A" else np.arctan2(v[0], -v[1])   # joint 1 faces the goal
    seeds = list(seeds) + [np.array([j1, -0.35, -0.75, 0.0, 1.1, 0.0])]       # elbow up, tool down
    goals = [T_goal] + ([T_goal @ pose(np.diag([-1, -1, 1]), (0, 0, 0))] if turn_ok else [])
    found = [q for G in goals for s in seeds if (q := ik(cell, r, G, s)) is not None]
    if not found:
        raise RuntimeError(f"robot {r} cannot reach {np.round(T_goal[:3, 3], 3)}")
    return min(found, key=lambda q: np.linalg.norm(q - seeds[0]))


# --- robot program --------------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    q0: np.ndarray
    q1: np.ndarray
    speed: float = 1.0           # share of full speed, 0 = robot stands still
    seconds: float = 0.0         # how long a step without motion takes
    opening: float = 0.06        # gripper opening (m)
    carries: str = ""            # part in the gripper
    at_fixture: bool = False     # needs the fixture
    event: str = ""              # true once the step is done
    wait_for: tuple = ()         # events this step waits for
    jaws: tuple = ()             # (part, "end" or "side"): these jaws close now


def robot_program(cell, r):
    """All steps of robot r, and how it holds each part (part pose in the tool frame)."""
    s = 1 if r == "A" else -1
    (x1, y1), (x2, _) = RACK["Pos1"], RACK["Pos2"]
    q_park = solve(cell, r, tool_down(s * (x1 + x2) / 2, s * y1, RACK_Z + 0.20, np.pi / 2), [], turn_ok=True)
    q, steps, grasp = q_park, [], {}
    for i, part in enumerate(JOBS[r]):
        kind = part[:4]
        _, width, height = PROFILE[kind]
        x, y, z, yaw = FIXTURE_POSE[part]
        rx, ry = RACK[kind]
        T_in_rack = part_pose(s * rx, s * ry, RACK_Z, np.pi / 2)
        # I plan the place pose first, then the pick, so the grip is the same
        q_place = solve(cell, r, tool_down(x, y, z + height - GRASP_DEPTH, yaw), [q], turn_ok=True)
        T_place = cell.fk(r, q_place)
        grasp[part] = np.linalg.inv(T_place) @ part_pose(x, y, z, yaw)
        T_pick = T_in_rack @ np.linalg.inv(grasp[part])
        q_pick = solve(cell, r, T_pick, [q_place, q])
        q_pick_up = solve(cell, r, lifted(T_pick), [q_pick])
        q_place_up = solve(cell, r, lifted(T_place), [q_place])
        wide, closed = width + 0.02, width
        steps += [
            Step(f"{part}: to the rack", q, q_pick_up, opening=wide, at_fixture=i > 0),
            Step(f"{part}: down to the part", q_pick_up, q_pick, SLOW, opening=wide),
            Step(f"{part}: grip", q_pick, q_pick, 0, T_GRIP, closed, event=f"gripped {part}"),
            Step(f"{part}: lift", q_pick, q_pick_up, SLOW, opening=closed, carries=part),
            Step(f"{part}: to the fixture", q_pick_up, q_place_up, opening=closed, carries=part, at_fixture=True,
                 wait_for=("located Pos1_a", "located Pos1_b") if kind == "Pos2" else ()),
            Step(f"{part}: set down", q_place_up, q_place, SLOW, opening=closed, carries=part, at_fixture=True,
                 event=f"set down {part}"),
            Step(f"{part}: open gripper", q_place, q_place, 0, T_OPEN, closed + 0.004, at_fixture=True),
            # end jaws first (low force, the part can still slide), then side jaws
            Step(f"{part}: end jaws close", q_place, q_place, 0, T_JAWS, closed + 0.004, at_fixture=True,
                 jaws=(part, "end"), event=f"end jaws closed {part}"),
            Step(f"{part}: side jaws close", q_place, q_place, 0, T_JAWS, closed + 0.004, at_fixture=True,
                 jaws=(part, "side"), event=f"located {part}"),
            Step(f"{part}: up", q_place, q_place_up, SLOW, opening=wide, at_fixture=True),
        ]
        q = q_place_up
    steps.append(Step("back to park", q, q_park, at_fixture=True))
    return steps, grasp


def move_profile(step):
    """Point-to-point move with a trapezoid speed profile, like the robot controller does it.
    Returns the duration and s(t), which runs from 0 to 1."""
    dist = np.max(np.abs(step.q1 - step.q0))
    if step.speed == 0 or dist < 1e-9:                # no motion: s is just the time gone
        T = step.seconds
        return T, (lambda t: min(max(t / T, 0.0), 1.0)) if T > 0 else (lambda t: 1.0)
    v, a = QD_MAX * step.speed / dist, QDD_MAX * step.speed / dist
    ta = min(v / a, np.sqrt(1 / a))               # time to speed up
    vp = a * ta                                   # top speed
    T = 1 / vp + ta

    def s(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            return 0.5 * a * t * t
        if t > T - ta:
            return 1.0 - 0.5 * a * (T - t) ** 2
        return 0.5 * a * ta * ta + vp * (t - ta)
    return T, s


# --- sequence -------------------------------------------------------------------------------------
@dataclass
class Frame:
    t: float
    qpos: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    state: dict                  # robot -> moving, gripper, waiting or done
    fixture: str                 # robot at the fixture, or ""
    carried: dict                # robot -> part in its gripper, or ""
    doing: dict                  # robot -> what it does, in words


def part_name(part):
    """Pos1_a -> Pos 1a"""
    return f"Pos {part[3]}{part[5]}"


def in_words(step):
    if ": " not in step.name:
        return "goes back to its rack"
    part, what = step.name.split(": ")
    p = part_name(part)
    return {"to the rack": f"goes to the rack for {p}", "down to the part": f"moves down to {p}",
            "grip": f"grips {p}", "lift": f"lifts {p}", "to the fixture": f"carries {p} to the fixture",
            "set down": f"sets {p} down", "open gripper": "opens the gripper",
            "end jaws close": "closes the end jaws", "side jaws close": "closes the side jaws",
            "up": "moves up"}[what]


def run_sequence(cell):
    """Runs both robots with the fixture rule. Returns the frames and the event log."""
    programs = {r: robot_program(cell, r) for r in "AB"}
    grasp = {**programs["A"][1], **programs["B"][1]}
    steps = {r: programs[r][0] for r in "AB"}
    rack = {}
    for r in "AB":
        s = 1 if r == "A" else -1
        for part in JOBS[r]:
            rx, ry = RACK[part[:4]]
            rack[part] = part_pose(s * rx, s * ry, RACK_Z, np.pi / 2) @ along(OFFSET[part])

    idx = {r: 0 for r in "AB"}               # step each robot is on
    started = {r: None for r in "AB"}        # when that step started
    profile = {r: None for r in "AB"}
    s_now = {r: 0.0 for r in "AB"}
    done = {r: False for r in "AB"}
    events, log, frames = set(), [], []
    holder, queue = "", []                   # robot at the fixture, robots waiting for it
    t = 0.0
    while not all(done.values()) and t < 120:
        state, doing = {}, {}
        for r in "AB":
            if done[r]:
                state[r], doing[r] = "done", "has finished"
                continue
            step = steps[r][idx[r]]
            if started[r] is None:           # can the step start?
                if not set(step.wait_for) <= events:
                    missing = [part_name(e.split()[-1]) for e in step.wait_for if e not in events]
                    state[r] = "waiting"
                    doing[r] = f"waits until {' and '.join(missing)} {'are' if len(missing) > 1 else 'is'} in place"
                    continue
                if step.at_fixture and holder != r:
                    if r not in queue:
                        queue.append(r)
                    if holder or queue[0] != r:      # first come, first served; robot 1 wins a tie
                        state[r], doing[r] = "waiting", "waits for the fixture"
                        continue
                    holder = queue.pop(0)
                started[r], profile[r] = t, move_profile(step)
            T, s_of = profile[r]
            tau = t + DT - started[r]
            s_now[r] = s_of(tau)
            state[r] = "moving" if step.speed > 0 else "gripper"
            doing[r] = in_words(step)
            if tau >= T - 1e-9:              # step done
                if step.event:
                    events.add(step.event)
                    log.append((round(t + DT, 2), r, step.event))
                nxt = steps[r][idx[r] + 1] if idx[r] + 1 < len(steps[r]) else None
                if holder == r and (nxt is None or not nxt.at_fixture):
                    holder = ""              # back at the rack, fixture is free again
                if nxt is None:
                    done[r] = True
                else:
                    idx[r], started[r], s_now[r] = idx[r] + 1, None, 0.0
        t += DT
        frames.append(record(cell, t, steps, idx, s_now, events, grasp, rack, state, holder, doing))
    return frames, log


def record(cell, t, steps, idx, s_now, events, grasp, rack, state, holder, doing):
    """Moves robots, jaws and parts to time t and saves a frame."""
    carried, closing = {}, {}
    for r in "AB":
        step = steps[r][idx[r]]
        cell.set_robot(r, step.q0 + s_now[r] * (step.q1 - step.q0), step.opening)
        carried[r] = step.carries
        if step.jaws:
            closing[step.jaws] = s_now[r]
    shut = {}                                    # 0 = open, 1 = closed
    for part in FIXTURE_POSE:
        for group, done in (("end", f"end jaws closed {part}"), ("side", f"located {part}")):
            shut[part, group] = 1.0 if done in events else closing.get((part, group), 0.0)
            cell.set_jaws(part, group, shut[part, group])
    mujoco.mj_kinematics(cell.model, cell.data)
    for part in FIXTURE_POSE:
        if f"set down {part}" in events:         # the end jaws push the part into place
            gap = JAW_OPEN * (1 - shut[part, "end"])
            cell.set_part(part, part_pose(*FIXTURE_POSE[part]) @ along(np.clip(OFFSET[part], -gap, gap)))
        elif f"gripped {part}" in events:
            r = "A" if part in JOBS["A"] else "B"
            cell.set_part(part, cell.tcp(r) @ grasp[part] @ along(OFFSET[part]))
        else:
            cell.set_part(part, rack[part])
    d = cell.data
    return Frame(round(t, 3), d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), state, holder, carried, doing)


def closest_distance(cell, frames):
    """Smallest gap between the two robots, with the parts they carry. This needs MuJoCo 3.3 or
    newer: with 3.2 I got wrong distances between boxes."""
    m = cell.model
    names = [m.geom(g).name for g in range(m.ngeom)]
    fromto = np.zeros(6)
    best = 1.0
    for f in frames:
        cell.show(f)
        sets = {r: [g for g, n in enumerate(names) if n.startswith(r + "_") or (f.carried[r] and n.startswith(f.carried[r]))]
                for r in "AB"}
        for ga in sets["A"]:
            for gb in sets["B"]:
                best = min(best, mujoco.mj_geomDistance(m, cell.data, ga, gb, best, fromto))
    return best


# --- 3D view --------------------------------------------------------------------------------------
def show_3d(cell, frames, speed=1.0):
    """Plays the frames in the MuJoCo viewer, again and again. Space pauses."""
    import mujoco.viewer
    paused = [False]

    def on_key(key):
        if key == 32:                                   # space
            paused[0] = not paused[0]

    with mujoco.viewer.launch_passive(cell.model, cell.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.10)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 180.0, -38.0
        t, last = 0.0, time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            if not paused[0]:
                t += (now - last) * speed
            last = now
            if t > frames[-1].t + 2.0:                  # show the finished frame for 2 s, then start over
                t = 0.0
            f = frames[min(int(t / DT), len(frames) - 1)]
            with viewer.lock():
                cell.show(f)
            if hasattr(viewer, "set_texts"):            # text box, MuJoCo 3.3 and newer
                fixture = f"in use by {NAME[f.fixture]}" if f.fixture else "free"
                viewer.set_texts([(int(mujoco.mjtFontScale.mjFONTSCALE_150), int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                                   "Time\nRobot 1 (blue)\nRobot 2 (pink)\nFixture",
                                   f"{f.t:.1f} s\n{f.doing['A']}\n{f.doing['B']}\n{fixture}")])
            viewer.sync()
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Two xArm6 robots build the frame of drawing AMZ-SA-26-001")
    parser.add_argument("--no-view", action="store_true", help="only print the numbers")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    args = parser.parse_args()

    cell = Cell()
    frames, log = run_sequence(cell)
    located = [(t, e.split()[-1]) for t, _, e in log if e.startswith("located")]
    print("parts in place: " + ", ".join(f"{part_name(p)} {t:.1f} s" for t, p in located))
    print(f"both robots back at their racks: {frames[-1].t:.1f} s")
    for r in "AB":
        wait = sum(f.state[r] == "waiting" for f in frames) * DT
        print(f"{NAME[r]} waits {wait:.1f} s")
    cell.show(frames[-1])
    off = max(np.linalg.norm(cell.data.mocap_pos[cell.mocap[p]] - FIXTURE_POSE[p][:3]) for p in FIXTURE_POSE)
    print(f"parts set down up to {1000 * max(abs(o) for o in OFFSET.values()):.1f} mm off, "
          f"after the jaws {1000 * off:.3f} mm")
    print(f"closest distance between the robots: {1000 * closest_distance(cell, frames):.0f} mm")
    if not args.no_view:
        print("3D view: Space pauses, close the window to stop")
        show_3d(cell, frames, args.speed)


if __name__ == "__main__":
    main()
