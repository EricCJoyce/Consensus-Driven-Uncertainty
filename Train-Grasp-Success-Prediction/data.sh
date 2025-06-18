ESTIMATORS='EPOS GDRNPP ZebraPose'
GRIPPERS='Parallel Underactuated'
BOP_OBJECTS='LMO1 LMO5 LMO6 LMO8 LMO9 LMO10 LMO11 LMO12 YCBV2 YCBV3 YCBV4 YCBV5 YCBV8 YCBV9 YCBV12'
MODE='t3r3'                                                         #  Signed component-wise difference of translations; signed difference of 3 angles.
TRAINING_PORTION=0.8

#####################################################################  UNZIP DATA
if [ ! -d 'Results' ]                                               #  Unzip pose estimate data.
then
    mkdir Results
    mkdir Results/Pose-Estimates
    unzip -d Results/Pose-Estimates DOPE-Pose-Estimates.zip
    unzip -d Results/Pose-Estimates NCF-Pose-Estimates.zip
    unzip -d Results/Pose-Estimates EPOS-Pose-Estimates.zip
    unzip -d Results/Pose-Estimates GDRNPP-Pose-Estimates.zip
    unzip -d Results/Pose-Estimates ZebraPose-Pose-Estimates.zip
    unzip -d Results/Parallel-Gripper Parallel-Gripper.zip
    unzip -d Results/Underactuated-Gripper Underactuated-Gripper.zip
fi

#####################################################################  ORGANIZE
echo 'COLLECTING ESTIMATOR POSE DATA'
echo ''

for estimator in $ESTIMATORS                                        #  Collect 6-DoF estimator pose estimates.
do
    if [ ! -f 'dataset-'$estimator'.txt' ]
    then
        python3 collect_pose_data.py -est $estimator -dat YCBV -dat LMO -v
        echo ''
    fi
done

#####################################################################  COMPUTE PRINCIPAL-SUPPORTER DIFFERENCES
echo 'COMPUTING A-B A-C DIFFERENCES for '$MODE
echo ''
                                                                    #  Derive A-B A-C datasets, each estimator acting in turn as PRINCIPAL.
if [ ! -f 'dataset-EPOS-'$MODE'.txt' ]
then
    python3 abac.py -principal EPOS -support GDRNPP -support ZebraPose -mode $MODE -v
    echo ''
fi
if [ ! -f 'dataset-GDRNPP-'$MODE'.txt' ]
then
    python3 abac.py -principal GDRNPP -support EPOS -support ZebraPose -mode $MODE -v
    echo ''
fi
if [ ! -f 'dataset-ZebraPose-'$MODE'.txt' ]
then
    python3 abac.py -principal ZebraPose -support EPOS -support GDRNPP -mode $MODE -v
    echo ''
fi

#####################################################################  COMPUTE PRINCIPAL-SUPPORTER DIFFERENCES
echo 'COMPUTING A-B A-C DIFFERENCES for ADD'
echo ''

if [ ! -f 'dataset-EPOS-add.txt' ]
then
    python3 abac.py -principal EPOS -support GDRNPP -support ZebraPose -mode add -v
    echo ''
fi
if [ ! -f 'dataset-GDRNPP-add.txt' ]
then
    python3 abac.py -principal GDRNPP -support EPOS -support ZebraPose -mode add -v
    echo ''
fi
if [ ! -f 'dataset-ZebraPose-add.txt' ]
then
    python3 abac.py -principal ZebraPose -support EPOS -support GDRNPP -mode add -v
    echo ''
fi

#####################################################################  PARTITION
echo 'PARTITIONING into TRAINING and TEST SETS for EACH OBJECT, EACH GRIPPER'
echo ''

for estimator in $ESTIMATORS                                        #  Each estimator, in turn, as PRINCIPAL.
do
    for gripper in $GRIPPERS                                        #  Separate grippers.
    do
        for bop_object in $BOP_OBJECTS                              #  Separate objects.
        do
            if [ ! -f 'train-'$estimator'-'$bop_object'-'$gripper'-'$MODE'.txt' ]
            then                                                    #  Define the split.
                python3 partition_datasets.py -principal $estimator -train $TRAINING_PORTION -object $bop_object -gripper $gripper -mode $MODE -shuffle
            fi
        done
    done
done

if [ ! -f 'data-split-profile.txt' ]                                #  Save a report of the data split.
then
    python3 profile_data_split.py
fi

#####################################################################  CLONE the SPLIT to the BASELINE INPUT TYPE (ADD).
echo 'CLONING PARTITION from '$MODE' to ADD (baseline)'
echo ''

for estimator in $ESTIMATORS                                        #  Clone the partitioned data sets so that they point to the ADD samples.
do
    for gripper in $GRIPPERS
    do
        for bop_object in $BOP_OBJECTS
        do
            if [ ! -f 'train-'$estimator'-'$bop_object'-'$gripper'-add.txt' ]
            then
                python3 clone_to.py -file 'train-'$estimator'-'$bop_object'-'$gripper'-'$MODE'.txt' -src $MODE -dst add
            fi

            if [ ! -f 'test-'$estimator'-'$bop_object'-'$gripper'-add.txt' ]
            then
                python3 clone_to.py -file 'test-'$estimator'-'$bop_object'-'$gripper'-'$MODE'.txt' -src $MODE -dst add
            fi
        done
    done
done

#####################################################################  ORGANIZE.

if [ ! -d '1Obj1Grip' ]
then
    mkdir 1Obj1Grip
fi

if [ ! -d '1ObjNGrip' ]
then
    mkdir 1ObjNGrip
fi

if [ ! -d 'Baseline' ]
then
    mkdir Baseline
fi

if [ ! -d 'DataSplit' ]
then
    mkdir DataSplit
fi

if [ ! -d 'DataSplit/'$MODE ]
then
    mkdir 'DataSplit/'$MODE
fi

if [ ! -d 'DataSplit/add' ]
then
    mkdir DataSplit/add
fi

for estimator in $ESTIMATORS
do
    mv 'dataset-'$estimator'-'$MODE'.npz' 'DataSplit/'$MODE
    mv 'dataset-'$estimator'-'$MODE'.txt' 'DataSplit/'$MODE
    mv 'train-'$estimator'-'*'-'$MODE'.txt' 'DataSplit/'$MODE
    mv 'test-'$estimator'-'*'-'$MODE'.txt' 'DataSplit/'$MODE

    mv 'dataset-'$estimator'-add.npz' DataSplit/add
    mv 'dataset-'$estimator'-add.txt' DataSplit/add
    mv 'train-'$estimator'-'*'-add.txt' DataSplit/add
    mv 'test-'$estimator'-'*'-add.txt' DataSplit/add
done
