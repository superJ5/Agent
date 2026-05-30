# Camera Installation Guide

## Camera Overview

This is a networked camera that is exceptionally simple to deploy and configure due to its integration into the dashboard and the use of cloud augmented edge storage. The MV family eliminates the complex and costly servers and video recorders required by traditional solutions which removes the limitations typically placed on video surveillance deployments.

### Package Contents

In addition to the MV camera, the following are provided:


<PIC:Security_Camera_01>


<PIC:Security_Camera_02>


## Pre-Install Preparation

You should complete the following steps before going on-site to perform an installation.

### Configure Your Network in Dashboard

The following is a brief overview only of the steps required to add a camera to your network. For detailed instructions about creating, configuring and managing Camera networks, refer to the online documentation. 1. Login to website. If this is your first time, create a new account. 2. Find the network to which you plan to add your cameras or create a new network. 3. Add your cameras to your network. You will need your order number (found on your invoice) or the serial number of each camera, which looks like Qxxx-xxxx-xxxx, and is found on the bottom of the unit. 4. Verify that the camera is now listed under Cameras > Monitor > Cameras.

#### Check and Configure Firewall Settings

If a firewall is in place, it must allow outgoing connections on particular ports to particular IP addresses. The most current list of outbound ports and IP addresses for your particular organization can be found here.

### DNS Configuration

Each camera will generate a unique domain name to allow for secured direct streaming functionality. These domain names resolve an A record for the private IP address of the camera. Any public recursive DNS server will resolve this domain. If utilizing an on site DNS server, please whitelist *.devices.meraki.direct or configure a conditional forwarder so that local domains are not appended to *.devices.meraki.direct and that these domain requests are forwarded.

### Assigning IP Addresses

At this time, the camera does not support static IP assignment. camera units must be added to a subnet that uses DHCP and has available DHCP addresses to operate correctly.

## Installation

Instructions Note: Each camera comes with an instruction pamphlet within the box. This pamphlet contains detailed step by step guides and images to assist in the physical install of the camera. A pdf of the pamphlet can be found here. Note: During first time setup, the camera will automatically update to the latest stable firmware. Some features may be unavailable until this automatic update is completed. This process may take up to 20 minutes due to enabling of whole disk encryption.

### Wall Mounting Instructions

For most mounting scenarios, the wall mount provides a quick, simple, and flexible means of mounting your device. The installation should be done in a few simple steps: 1. Leave protective plastic sticker on camera bubble


<PIC:Manual33_0>


2. Use template to determine mounting hole locations before screwing in the mount plate. Peel backing from mount template to stick on wall. Slide the camera onto the wall mount.


<PIC:Manual33_1>


3. Screw the mounting plate onto the wall in pre-determined locations. Use template holes marked with the letter "A" for standard wall mounting.


<PIC:Manual33_2>


4. Connect PoE cable to camera. For cords that will exit the top of the camera, loop the cable inside the camera as shown.


<PIC:Manual33_3>


5. Slide camera over top of mount plate and slide down into mount plate hooks. Secure with safety screw.


<PIC:Manual33_4>


6. Turn bubble counter clockwise to unlock. Hinge bubble off of body to remove.


<PIC:Manual33_5>


7. Pinch near thumb screws and pull straight away from the camera to remove lens guard.


<PIC:Manual33_6>


8. Aim the lens. Look through the camera on the Dashboard to fine-tune the picture. The camera sensor and lens unit can be physically tilted through a range of 65 degrees, rotated through a range of 350 degrees, and panned through a range of 350 degrees. The image can only be rotated by 180 degrees in software and no other adjustments can be made. Zoom and focus can be adjusted remotely and cannot be adjusted physically on the camera.


<PIC:Manual33_7>


9. Replace lens guard and bubble. Turn bubble clockwise to lock.


<PIC:Manual33_8>


10. Remove protective plastic sticker. Check LED function. Use the Dashboard to adjust camera focus and configure other settings.


<PIC:Manual33_9>


### T-rail Mounting Instructions

To mount your camera on a drop ceiling T-rail, use the included hardware. The hardware can be used to mount to most 9/16", 15/16", or 1 1/2" T-rails.

1. Using the dashed lines on the mount plate template as a guide, set the proper spacing of the clips.


<PIC:Manual33_11>


2. Tighten the set screws on the T-rail clips and secure them using a 5/64" (2 mm) hex key.


<PIC:Manual33_12>


3. Attach the mount plate to the T-rail clips using the mount plate holes (marked with a "G").


<PIC:Manual33_13>


4. Attach the T-rail clips to the T-rail by rotating them and snapping them into place as shown. The black foam pads should be compressed slightly after installation.


<PIC:Manual33_10>


### Modifications for recessed T-rail

Standard 6-32x4mm T-rail screws


<PIC:Manual33_14>




## Powering the camera

Remove the cable guard and route the Ethernet cable from an active port on an 802.3af PoE switch or PoE injector.

Note: Power over Ethernet supports a maximum cable length of 300 ft (100 m).

## LED Indicator

Your camera is equipped with a LED light on the front of the unit to convey information about system functionality and performance: Flashing Green (2 second interval) - MV is upgrading or initializing for the first time. Solid Green - MV is operating nominally.

## 2. 图片锚点顺序

- `<PIC:Security_Camera_01>`
- `<PIC:Security_Camera_02>`
- `<PIC:Manual33_0>`
- `<PIC:Manual33_1>`
- `<PIC:Manual33_2>`
- `<PIC:Manual33_3>`
- `<PIC:Manual33_4>`
- `<PIC:Manual33_5>`
- `<PIC:Manual33_6>`
- `<PIC:Manual33_7>`
- `<PIC:Manual33_8>`
- `<PIC:Manual33_9>`
- `<PIC:Manual33_11>`
- `<PIC:Manual33_12>`
- `<PIC:Manual33_13>`
- `<PIC:Manual33_10>`
- `<PIC:Manual33_14>`

## 3. 图片路径对照

### <PIC:Security_Camera_01>

- image_id：`Security_Camera_01`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Security_Camera_01.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Security_Camera_01.jpg`

### <PIC:Security_Camera_02>

- image_id：`Security_Camera_02`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Security_Camera_02.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Security_Camera_02.jpg`

### <PIC:Manual33_0>

- image_id：`Manual33_0`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_0.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_0.jpg`

### <PIC:Manual33_1>

- image_id：`Manual33_1`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_1.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_1.jpg`

### <PIC:Manual33_2>

- image_id：`Manual33_2`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_2.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_2.jpg`

### <PIC:Manual33_3>

- image_id：`Manual33_3`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_3.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_3.jpg`

### <PIC:Manual33_4>

- image_id：`Manual33_4`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_4.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_4.jpg`

### <PIC:Manual33_5>

- image_id：`Manual33_5`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_5.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_5.jpg`

### <PIC:Manual33_6>

- image_id：`Manual33_6`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_6.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_6.jpg`

### <PIC:Manual33_7>

- image_id：`Manual33_7`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_7.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_7.jpg`

### <PIC:Manual33_8>

- image_id：`Manual33_8`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_8.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_8.jpg`

### <PIC:Manual33_9>

- image_id：`Manual33_9`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_9.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_9.jpg`

### <PIC:Manual33_11>

- image_id：`Manual33_11`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_11.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_11.jpg`

### <PIC:Manual33_12>

- image_id：`Manual33_12`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_12.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_12.jpg`

### <PIC:Manual33_13>

- image_id：`Manual33_13`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_13.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_13.jpg`

### <PIC:Manual33_10>

- image_id：`Manual33_10`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_10.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_10.jpg`

### <PIC:Manual33_14>

- image_id：`Manual33_14`
- 相对路径：`data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_14.jpg`
- 绝对路径：`E:\.codex\worktrees\3f3c\ai_agent_competition\data\manuals\raw\13_网络摄像机_Security_Camera\images\Manual33_14.jpg`
