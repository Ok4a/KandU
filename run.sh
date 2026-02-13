config_folder=`ls ./config/*.ini`
for config_file in $config_folder
do
   echo $config_file
   python3 percentParShiftLaplace.py $config_file
done