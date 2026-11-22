"""Generate a Gazebo Fortress world from a project course seed."""

import argparse
import os
from pathlib import Path
from sim import ALTITUDE, FAR, HEIGHT, HFOV, WIDTH, course


def geometry_box(w, d, h):
    return f'<geometry><box><size>{w} {d} {h}</size></box></geometry>'


def visual(name, geometry, rgb, pose='0 0 0 0 0 0'):
    color = ' '.join(str(c / 255) for c in rgb) + ' 1'
    return (f'<visual name="{name}"><pose>{pose}</pose>{geometry}'
            f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>')


def shape(name, box, rgba=None):
    box_geom = geometry_box(box.w, box.d, box.h)
    return (f'<model name="{name}"><static>true</static>'
            f'<pose>{box.x} {box.y} {box.z + box.h/2} 0 0 0</pose>'
            f'<link name="static"><collision name="body">{box_geom}</collision>'
            f'{visual("body", box_geom, rgba or box.color)}</link></model>')


def generate(layout, rate=10):
    from sim import Box
    real_time_factor = float(os.environ.get('GAZEBO_REAL_TIME_FACTOR', '1'))
    plugins = ''.join(f'<plugin filename="ignition-gazebo-{library}-system" '
                      f'name="gz::sim::systems::{name}">{body}</plugin>'
                      for library, name, body in [
                          ('physics', 'Physics', ''), ('scene-broadcaster', 'SceneBroadcaster', ''),
                          ('user-commands', 'UserCommands', ''),
                          ('sensors', 'Sensors', '<render_engine>ogre2</render_engine>')])
    gyro_visuals = visual('body', geometry_box(.26, .21, .08), (33, 59, 66))
    for idx, (x, y) in enumerate(((-.17, -.17), (-.17, .17), (.17, -.17), (.17, .17))):
        rotor = '<geometry><cylinder><radius>.09</radius><length>.024</length></cylinder></geometry>'
        gyro_visuals += visual(f'rotor_{idx}', rotor, (43, 108, 107), f'{x} {y} .04 0 0 0')
    body = geometry_box(.39, .39, .11)
    ground = shape('floor', Box(8, 0, 160, 160, .2, (118, 125, 109), -.20))
    obstacles = ''.join(shape(f'obstacle_{i}', b) for i, b in enumerate(layout.boxes))
    return f'''<?xml version="1.0"?>
<sdf version="1.8">
  <world name="course">
    <physics name="step" type="ignored"><max_step_size>.004</max_step_size><real_time_factor>{real_time_factor}</real_time_factor></physics>
    {plugins}
    <scene><ambient>.55 .55 .55 1</ambient><background>.50 .65 .78 1</background></scene>
    <light type="directional" name="sun">
      <pose>0 0 8 0 0 0</pose><diffuse>.85 .85 .81 1</diffuse><specular>.15 .15 .15 1</specular>
      <direction>-.4 .2 -1</direction><cast_shadows>true</cast_shadows>
    </light>
    {ground}
    {obstacles}
    <model name="drone">
      <pose>0 0 {ALTITUDE} 0 0 0</pose>
      <link name="body">
        <gravity>false</gravity>
        <inertial><mass>1</mass><inertia><ixx>.015</ixx><iyy>.015</iyy><izz>.025</izz></inertia></inertial>
        <collision name="drone_body">{body}</collision>
        {gyro_visuals}
        <sensor name="rgbd" type="rgbd_camera">
          <pose>.23 0 0 0 0 0</pose>
          <topic>/drone/camera</topic><always_on>true</always_on><update_rate>{rate}</update_rate>
          <camera><horizontal_fov>{HFOV}</horizontal_fov>
            <image><width>{WIDTH}</width><height>{HEIGHT}</height><format>R8G8B8</format></image>
            <clip><near>.05</near><far>{FAR}</far></clip>
          </camera>
        </sensor>
      </link>
      <plugin filename="ignition-gazebo-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <topic>/drone/cmd_vel</topic>
      </plugin>
      <plugin filename="ignition-gazebo-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_topic>/drone/odom</odom_topic><odom_publish_frequency>50</odom_publish_frequency><dimensions>3</dimensions>
      </plugin>
    </model>
  </world>
</sdf>
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int)
    parser.add_argument('out', type=Path)
    parser.add_argument('--safety', action='store_true')
    args = parser.parse_args()
    args.out.write_text(generate(course(args.seed, safety=args.safety)), encoding='utf-8')
