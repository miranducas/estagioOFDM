val_ant=0
for val in 0 5 10 15 20 25 30 35 40; do
   sed -i 's/SNRdb='$val_ant'/SNRdb='$val'/' Main_ml.py   
   val_ant=$val
   python3 Main_ml.py   
   
done