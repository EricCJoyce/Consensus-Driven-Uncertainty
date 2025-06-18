ESTIMATORS='EPOS GDRNPP ZebraPose'
GRIPPERS='Parallel Underactuated'
BOP_OBJECTS='LMO1 LMO5 LMO6 LMO8 LMO9 LMO10 LMO11 LMO12 YCBV2 YCBV3 YCBV4 YCBV5 YCBV8 YCBV9 YCBV12'
#BOP_OBJECTS='LMO12 YCBV4 YCBV9 YCBV12'
GRIP_SET_NAME='both'
EXPERIMENT_NUMBER=0                                                 #  However you named this folder in GitHub.
MODE='t3r3'                                                         #  30sep24: by decree, we are back on t3r3. Hope it doesn't suck no that we use cosine annealing.
EPOCHS=3000
SCHED='cos'                                                         #  Cosine annealing.
LR=0.001
LR_MIN=0.00001

#####################################################################  TRAIN

for estimator in $ESTIMATORS                                        #  For all estimators...
do
    if [ ! -d $estimator ]                                          #  If no folder for this estimator exists, create one.
    then
        mkdir $estimator
    fi

    if [ ! -d $estimator'/'$GRIP_SET_NAME ]                         #  If no folder for this gripper-group exists, create one.
    then
        mkdir $estimator'/'$GRIP_SET_NAME
    fi

    for object in $BOP_OBJECTS                                      #  For all objects...
    do
        if [ ! -d $estimator'/'$GRIP_SET_NAME'/'$object ]           #  If no folder for this object exists, create one.
        then
            mkdir $estimator'/'$GRIP_SET_NAME'/'$object
        fi

        already_trained=true
        for dir in $estimator'/'$GRIP_SET_NAME'/'$object'/pth' $estimator'/'$GRIP_SET_NAME'/'$object'/AUC' $estimator'/'$GRIP_SET_NAME'/'$object'/bookmarks' $estimator'/'$GRIP_SET_NAME'/'$object'/confusion-matrices' $estimator'/'$GRIP_SET_NAME'/'$object'/records'
        do
            if ! [ -d "$dir" ]
            then
                already_trained=false
                break
            fi
        done

        #############################################################  Train network, save and store related files.

        if [ "$already_trained" = false ]
        then
            echo 'TRAINING '$estimator': '$object', '$GRIP_SET_NAME
            echo ''

            gripper_str=''

            for gripper in $GRIPPERS
            do
                gripper_str=$gripper_str' -gripper '$gripper
            done
                                                                    #  Train a network for this (principal, object, gripper-set).
            python3 train_1ObjNGrip.py -principal $estimator -gripname $GRIP_SET_NAME$gripper_str -obj $object -epochs $EPOCHS -optim Adam -lrsched $SCHED -lr $LR -lrmin $LR_MIN -batch 16 -mode $MODE -v -plot -report -save none

            mkdir $estimator'/'$GRIP_SET_NAME'/'$object'/pth'
            mkdir $estimator'/'$GRIP_SET_NAME'/'$object'/bookmarks'
            mkdir $estimator'/'$GRIP_SET_NAME'/'$object'/records'
            mkdir $estimator'/'$GRIP_SET_NAME'/'$object'/confusion-matrices'
            mkdir $estimator'/'$GRIP_SET_NAME'/'$object'/AUC'
                                                                    #  Save the best records to the principal-gripper-object root.
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'$MODE'-best-confusionmatrix.txt' $estimator'/'$GRIP_SET_NAME'/'$object
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'$MODE'-best-record.txt' $estimator'/'$GRIP_SET_NAME'/'$object

            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'$MODE'-bestGeneralizing-confusionmatrix.txt' $estimator'/'$GRIP_SET_NAME'/'$object
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'$MODE'-bestGeneralizing-record.txt' $estimator'/'$GRIP_SET_NAME'/'$object
                                                                    #  Compute AUCs for every epoch.
            #for record in 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-record.txt'
            #do
            #    python3 auc_NObj1Grip.py -nn $record -f avg -v
            #done
                                                                    #  Save everything else to their respective folders.
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-confusionmatrix.txt' $estimator'/'$GRIP_SET_NAME'/'$object'/confusion-matrices'
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-record.txt' $estimator'/'$GRIP_SET_NAME'/'$object'/records'
            #mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-AUC.txt' $estimator'/'$GRIP_SET_NAME'/'$object'/AUC'
            #mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-CDF.png' $estimator'/'$GRIP_SET_NAME'/'$object'/AUC'
            #mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'.pth' $estimator'/'$GRIP_SET_NAME'/'$object'/pth'
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'.txt' $estimator'/'$GRIP_SET_NAME'/'$object'/bookmarks'

            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-loss.png' $estimator'/'$GRIP_SET_NAME'/'$object
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-trainingLoss.png' $estimator'/'$GRIP_SET_NAME'/'$object
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-accuracy.png' $estimator'/'$GRIP_SET_NAME'/'$object
            mv 'GraspSuccess-'$estimator'-'$object'-'$GRIP_SET_NAME'-'*'-lr.png' $estimator'/'$GRIP_SET_NAME'/'$object
        else
            echo '  Already trained '$estimator': '$object', '$GRIP_SET_NAME
            echo ''
        fi

        #########################################################  Report and analysis (requires all AUC files.)

        #python3 report_NObj1Grip.py -principal $estimator -objname $object -gripper $gripper -mode $MODE -epochs $EPOCHS -lr $LR -experiment $EXPERIMENT_NUMBER

        #mv $estimator'-'$object'-'$gripper'.md' $estimator'/'$gripper'/'$object
    done
done
