# Camera Installation Guide  # Camera Overview  This is a networked camera that is exceptionally simple to deploy and configure due to its integration into the dashboard and the use of cloud augmented edge storage. The MV family eliminates the complex and costly servers and video recorders required by traditional solutions which removes the limitations typically placed on video surveillance deployments.  # Package Contents  In addition to the MV camera, the following are provided:  

![Security_Camera_01](images/Security_Camera_01.jpg)



![Security_Camera_02](images/Security_Camera_02.jpg)

  
# Pre-lnstall Preparation  You should complete the following steps before going on-site to perform an installation.  # Configure Your Network in Dashboard  The following is a brief overview only of the steps required to add an camera to your network. For detailed instructions about creating,configuring and managing Camera networks,refer to the online documentation   1. Login to website, If this is your first time, create a new account. 2. Find the network to which you plan to add your cameras or create a new network. 3. Add your cameras to your network. You will need your order number (found on your invoice) or the serial number of each camera, which looks like Qxxx-xxxx-xxxx, and is found on the bottom of the unit 4.Verify thatyou thecamera is nowlisted under Cameras  $>$  Monitor  $>$  Cameras.  # Check and Configure Firewall Settings  If a firewall is in place, it must allow outgoing connections on particular ports to particular IP addresses. The most current list of outbound ports and IP addresses for your particular organization can be found here  # DNS Configuration  Each camera will generate a unique domain name to allow for secured direct streaming functionality. These domain names resolve an A record for the private ip address of the camera.Any public recursive DNS server will resolve this domain.  If utilizing an on site DNS server, please whitelist \*.devices.direct or configure a conditional forwarder so that local domains are not appended to \*.devices.meraki.direct and that these domain requests are forwarded.  # Assigning IP Addresses  At this time, the camera does not support static IP assignment. camera units must be added to a subnet that uses DHCP and has available DHCP addresses to operate correctly.  
# Installation Instructions  Note: Each camera comes with an instruction pamphlet within the box. This pamphlet contains detailed step by step guides and images to assist in the physical install of the camera. A pdf of the pamphlet can be found here  Note: During first time setup, the camera will automatically update to the latest stable firmware. Some features may be unavailable until this automatic update is completed. This process may take up to 20 minutes due to enabling of whole disk encryption  # Wall Mounting Instructions  For most mounting scenarios, the wall mount provides a quick, simple, and flexible means of mounting your device. The installation should be done in a few simple steps:  1.Leave protective plastic sticker on camera bubble  

![Manual33_0](images/Manual33_0.jpg)

  2. Use template to determine mounting hole locations before screwing in the mount plate. Peel backing from mount template to stick on wall.Slide the camera on to the wall mount.  


![Manual33_1](images/Manual33_1.jpg)

  3. Screw the mounting plate onto the wall in pre-determined locations. Use template holes marked with the letter "A" for standard wall mounting.  

![Manual33_2](images/Manual33_2.jpg)

  4. Connect PoE cable to camera. For cords that will exit the top of the camera, loop the cable inside the camera as shown.  

![Manual33_3](images/Manual33_3.jpg)

  
5. Slide camera over top of mount plate and slide down into mount plate hooks. Secure with safety screw.  

![Manual33_4](images/Manual33_4.jpg)

  6. Turn bubble counter clockwise to unlock. Hinge bubble off of body to remove.  

![Manual33_5](images/Manual33_5.jpg)

  7. Pinch near thumb screws and pull straight away from the camera to remove lens guard.  


![Manual33_6](images/Manual33_6.jpg)

  8.Aimthelens.Look through the camera on the Dashboard to fine tune the picture.The camera sensor and lens unit can be physically tilted through a range of 65 degrees, rotated through a range of 350 degrees, and panned through a range of 350degrees.The image can onlybe rotated by180 degrees insoftware and no other adjustments can be made. Zoom and focus can be adjusted remotely and can not be adjusted physically on the camera.  

![Manual33_7](images/Manual33_7.jpg)

  9.Replace lens guard and bubble.Turn bubble clockwise to lock  


![Manual33_8](images/Manual33_8.jpg)

  10. Remove protective plastic sticker. Check LED function. Use the Dashboard to adjust camera focus and configure other settings.  

![Manual33_9](images/Manual33_9.jpg)

  # T-rail Mounting Instructions  To mount your camera on a drop ceiling T-rail, use the included hardware. The hardware can be used to mount to most 9/ 16"，15/16",or11/2"T-rails.  1. Using the dashed lines on the mount plate template as a guide, set the proper spacing of the clips.  


![Manual33_10](images/Manual33_10.jpg)

  2. Tighten the set screws on the  $\mathsf{T}$  -rail clips and secure them using a  $5/64^{\circ}$   $(2\,\mathsf{m m})$  hexkey.  

![Manual33_11](images/Manual33_11.jpg)

  3. Attach the mount plate to the  $\mathsf{T}$  -rail clips using the mount plate holes (marked with a  $"G"$  》  


![Manual33_12](images/Manual33_12.jpg)

  4. Attach the T-rail clips to the T-rail by rotating them and snapping them into place as shown. The black foam pads should be compressed slightly after installation.  

![Manual33_13](images/Manual33_13.jpg)

  
Standard6-32x4mm T-railscrews  


![Manual33_14](images/Manual33_14.jpg)

  
Modifications for recessed T-rail  # Powering the camera Remove the cable guard and route the Ethernet cable from an active port on a 802.3af PoE switch or PoE injector  Note:Power over Ethernet supports a maximum cable length of 300 ft(100m)  # LED Indicator  Your camera is equipped with a LED light on the front of the unit to convey information about system functionality and performance:  Flashing Green (2 second interval) - MV is upgrading or initializing for the first time Solid Green - MV is operating nominally  
在·