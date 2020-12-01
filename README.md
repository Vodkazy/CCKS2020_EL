# CCKS2020_EL
Code for the game "CCKS 2020: 面向中文短文本的实体链指任务" , see https://www.biendata.xyz/competition/ccks_2020_el/

主要参照开源Baseline，https://blog.csdn.net/frank_zhaojianbo/article/details/106878950，思路是对于EntityLinking采用Roberta二分类，得分最高且超过阈值者作为EntityLinking的结果，否则转入EntityTyping，对于EntityTyping采用Roberta多分类，思路比较简单。
