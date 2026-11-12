# CI6 : Accélérer un ResNet

## Exercice 0 

La sortie de check_env.py est : </br>
Job 54598 sur arcadia-slurm-node-3 ; </br>
GPU : MIG-98abd536-0548-5116-b4f8-70a7163dd6fb ; </br>
CPU : 4 </br>
GPU 0: NVIDIA H100 NVL (UUID: GPU-c8a4c7b7-5b45-2033-9a40-92b4cf03cf33)</br>
MIG 1g.12gb    </br>
 Device  0: (UUID: MIG-98abd536-0548-5116-b4f8-70a7163dd6fb) </br>

La puissance de calcul d'un CPU est bien inférieure à celle d'un GPU. LES CPUs risquent d'avoir du mal à alimenter assez rapidement le H100 en mini-batchs, et donc toute la puissance du GPU ne sera pas exploitée, car le GPU va attendre. 