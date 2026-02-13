config_folder=`ls ./config/*.ini`
for config_file in $yourfilenames
do
   python percentParShiftLaplace.py $config_file
done