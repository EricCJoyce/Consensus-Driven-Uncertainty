ESTIMATORS='EPOS GDRNPP ZebraPose'
GRIPPERS='Parallel Underactuated'
BOP_OBJECTS='LMO1 LMO5 LMO6 LMO8 LMO9 LMO10 LMO11 LMO12 YCBV2 YCBV3 YCBV4 YCBV5 YCBV8 YCBV9 YCBV12'

#####################################################################  RUN BASELINE

for estimator in $ESTIMATORS                                        #  For all estimators...
do
    if [ ! -d $estimator ]                                          #  If no folder for this estimator exists, create one.
    then
        mkdir $estimator
    fi

    for gripper in $GRIPPERS                                        #  For all grippers...
    do
        if [ ! -d $estimator'/'$gripper ]                           #  If no folder for this estimator, this gripper exists, create one.
        then
            mkdir $estimator'/'$gripper
        fi

        for bop_object in $BOP_OBJECTS                              #  For all objects...
        do
            if [ ! -d $estimator'/'$gripper'/'$bop_object ]         #  If no folder for this estimator, this gripper exists, create one.
            then
                mkdir $estimator'/'$gripper'/'$bop_object
            fi

            echo 'RUNNING '$estimator': '$bop_object', '$gripper
            echo ''

            python3 baseline_1Obj1Grip.py -principal $estimator -object $bop_object -gripper $gripper

            python3 auc_1Obj1Grip.py -nn 'Baseline-'$estimator'-'$bop_object'-'$gripper'-record.txt' -f avg -v

            mkdir $estimator'/'$gripper'/'$bop_object'/records'
            mkdir $estimator'/'$gripper'/'$bop_object'/confusion-matrices'
            mkdir $estimator'/'$gripper'/'$bop_object'/AUC'
                                                                    #  Clean up working directory.
            mv 'Baseline-'$estimator'-'$bop_object'-'$gripper'-confusionmatrix.txt' $estimator'/'$gripper'/'$bop_object'/confusion-matrices'
            mv 'Baseline-'$estimator'-'$bop_object'-'$gripper'-record.txt' $estimator'/'$gripper'/'$bop_object'/records'
            mv 'Baseline-'$estimator'-'$bop_object'-'$gripper'-AUC.txt' $estimator'/'$gripper'/'$bop_object'/AUC'
            mv 'Baseline-'$estimator'-'$bop_object'-'$gripper'-CDF.png' $estimator'/'$gripper'/'$bop_object'/AUC'

            ########################################################  Report and analysis.

            python3 report_1Obj1Grip.py -principal $estimator -object $bop_object -gripper $gripper

            mv 'Baseline-'$estimator'-'$bop_object'-'$gripper'.md' $estimator'/'$gripper'/'$bop_object
        done
    done
done
