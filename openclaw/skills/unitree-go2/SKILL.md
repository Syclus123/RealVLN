---
name: unitree_go2
description: Control Unitree Go2 robot dog movements. Use when the user wants to control a Go2 robot dog to perform actions like stand up, move forward, flip, walk, or any sport mode commands.
---

# Unitree Go2 Controller

Control the Unitree Go2 robot dog using the sport client API.

## Usage

```bash
python scripts/go2_control.py <action_name> [network_interface] [--angle 角度]
```

If the robot doesn't respond, specify the network interface (e.g., `eth0`, `wlan0`):

## Available Actions

| Action | Description |
|--------|-------------|
| `damp` | Damp mode (relaxes joints) |
| `stand_up` | Stand up from ground |
| `stand_down` | Lie down |
| `move_forward` | Move forward |
| `move_backward` | Move backward |
| `move_lateral` | Move sideways |
| `turn_left` | Turn left (counter-clockwise) |
| `turn_right` | Turn right (clockwise) |
| `stop` | Stop all movement |
| `handstand` | Handstand pose (4 seconds) |
| `balance` | Balanced standing |
| `recovery` | Recovery stand |
| `left_flip` | Left side flip |
| `back_flip` | Back flip |
| `free_walk` | Free walk mode |
| `free_bound` | Free bound/jump mode (2 seconds) |
| `free_avoid` | Obstacle avoidance mode (2 seconds) |
| `walk_upright` | Upright walking (4 seconds) |
| `cross_step` | Cross-step walking (4 seconds) |
| `free_jump` | Jump mode (4 seconds) |
| `sit` | Sit down |
| `hello` | Greeting gesture |
| `stretch` | Stretch |

## Direction Control

### Turning with Custom Angle

Use `--angle` parameter to specify rotation angle (default: 90 degrees):

```bash
# Turn left 45 degrees
python scripts/go2_control.py turn_left eth0 --angle 45

# Turn right 180 degrees
python scripts/go2_control.py turn_right eth0 --angle 180

# Turn left 90 degrees (default)
python scripts/go2_control.py turn_left
```

**Note**: 
- `turn_left` - Rotate counter-clockwise
- `turn_right` - Rotate clockwise  
- Default angle is 90 degrees if not specified
- The old `move_rotate` action has been replaced with separate `turn_left` and `turn_right` commands

## Examples

Make the dog stand up:
```bash
python scripts/go2_control.py stand_up
```

Move forward:
```bash
python scripts/go2_control.py move_forward
```

Turn left 90 degrees:
```bash
python scripts/go2_control.py turn_left
```

Turn right 45 degrees:
```bash
python scripts/go2_control.py turn_right eth0 --angle 45
```

Do a back flip:
```bash
python scripts/go2_control.py back_flip
```

List all actions:
```bash
python scripts/go2_control.py list
```

## Safety Warning

⚠️ **Always ensure the area around the robot is clear before executing actions, especially flips and jumps.**

## Requirements

- unitree_sdk2py Python SDK must be installed
- Network connection to the robot must be established
