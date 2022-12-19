Event Driven Simulation:

---------------------------------------------------  
Elements 

---------------------------------------------------  

Class UE: 
Queues, Attributes (SINR, SINR2, ...)
    UE-Types:
    - rt-user: ON-OFF TIME -> ON-duration: neg. exp. distr. with mean 3s OFF-duration: neg. exp. distr. with mean=3s and upper limit of 6.9s
                During ON-time: 20Byte every 20ms
    - ftp-user: packet-arrival-time: neg. exp distr. with mean=180s (reading time) packet size: 2MByte
    - best effort: packet size as parameter & timeout 10ms
    - best effort_stat: packet size 4000ms & arrival time = neg. exp. dist. with mean= parameters
    - streaming user: 1.5 Mbps -> 3000 bit every 2 ms

Class Scheduler:
attributes: rem_prb={}, rem_req={}, rem_prb_c={}, rem_req_c={} 
-> saves for each scheduling round the remaining ressources (not used), and the remaining requests of users that could not been scheduled (all measured as prbs) -> rem_prb_c=123 -> 123 prbs would be needed to cover all requests
-> "c" stands for comp scheduler 

   Scheduler Types: 
   - central scheduler: scheduling for comp users -> for several sectors together
   parameter: users (all comp users), SCHEDULE_T (here set to 2ms every time), cluster (list of sectors from the cluster scheduled by the  central scheduler), prb_number (number of prb reserved for comp of each sector), sched_metric ([e1,e2] -> give the exponents for the metric)
   
   - scheduler: scheduling for each seperate sector -> each sector has on scheduling instance
   parameter: users (all users) , SCHEDULE_T, cluster, prb_number (prb_number for all users -> here mostly 50), users2 (all users except from comp users), prb_number2 (all prbs - comp prbs), sched_metric 
   
  
---------------------------------------------------  
Simulation procedure:

---------------------------------------------------

- define all paramters
- define environment
- define schedulers (N scheduler dor N sectors and 1 central scheduler per cluster)
- define index of which users to take (randomly or deterministic)
- restrict_users_to_cluster (users can only take sectors from the cluster for comp ) 
- get_user_from_cluster (select the users for the scheduling by the defined index)
- calculate the prb numbers for comp and non-comp users
- "set/put" users into the environment -> queue starts to fill 
- start the central scheduler and all the sector schedulers


    


