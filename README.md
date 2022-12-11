Event Driven Simulation:

Elements 

Class UE: 
Queues, Attributes (SINR, SINR2, ...)
    UE-Types:
    - rt-user: ON-OFF TIME -> ON-duration: neg. exp. distr. with mean 3s OFF-duration: neg. exp. distr. with mean=3s and upper limit of 6.9s
                During ON-time: 20Byte every 20ms
    - ftp-user: packet-arrival-time: neg. exp distr. with mean=180s (reading time) packet size: 2MByte
    - best effort: packet size as parameter & timeout 10ms
    - best effort_stat: packet size 4000ms & arrival time = neg. exp. dist. with mean= parameter
    - streaming user: 1.5 Mbps -> 3000 bit every 2 ms

Class Scheduler:

