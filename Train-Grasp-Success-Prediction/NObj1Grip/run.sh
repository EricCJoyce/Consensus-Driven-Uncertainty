ESTIMATORS='EPOS GDRNPP ZebraPose'
GRIPPERS='Parallel Underactuated'
BOP_OBJECTS='LMO1 LMO5 LMO6 LMO8 LMO9 LMO10 LMO11 LMO12 YCBV2 YCBV3 YCBV4 YCBV5 YCBV8 YCBV9 YCBV12'
#BOP_OBJECTS='LMO12 YCBV4 YCBV9 YCBV12'
OBJ_SET_NAME='all'
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

    for gripper in $GRIPPERS                                        #  For all grippers...
    do
        if [ ! -d $estimator'/'$gripper ]                           #  If no folder for this estimator, this gripper exists, create one.
        then
            mkdir $estimator'/'$gripper
        fi

        if [ ! -d $estimator'/'$gripper'/'$OBJ_SET_NAME ]           #  If no folder for this object-group exists, create one.
        then
            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME
        fi

        already_trained=true
        for dir in $estimator'/'$gripper'/'$OBJ_SET_NAME'/pth' $estimator'/'$gripper'/'$OBJ_SET_NAME'/AUC' $estimator'/'$gripper'/'$OBJ_SET_NAME'/bookmarks' $estimator'/'$gripper'/'$OBJ_SET_NAME'/confusion-matrices' $estimator'/'$gripper'/'$OBJ_SET_NAME'/records'
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
            echo 'TRAINING '$estimator': '$OBJ_SET_NAME', '$gripper
            echo ''

            obj_str=''

            for bop_object in $BOP_OBJECTS
            do
                obj_str=$obj_str' -obj '$bop_object
            done
                                                                    #  Train a network for this (principal, object, gripper).
            python3 train_NObj1Grip.py -principal $estimator -objname $OBJ_SET_NAME$obj_str -gripper $gripper -epochs $EPOCHS -optim Adam -lrsched $SCHED -lr $LR -lrmin $LR_MIN -batch 16 -mode $MODE -v -plot -report -save none

            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME'/pth'
            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME'/bookmarks'
            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME'/records'
            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME'/confusion-matrices'
            mkdir $estimator'/'$gripper'/'$OBJ_SET_NAME'/AUC'
                                                                    #  Save the best records to the principal-gripper-object root.
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'$MODE'-best-confusionmatrix.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'$MODE'-best-record.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME

            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'$MODE'-bestGeneralizing-confusionmatrix.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'$MODE'-bestGeneralizing-record.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME
                                                                    #  Compute AUCs for every epoch.
            #for record in 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-record.txt'
            #do
            #    python3 auc_NObj1Grip.py -nn $record -f avg -v
            #done
                                                                    #  Save everything else to their respective folders.
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-confusionmatrix.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME'/confusion-matrices'
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-record.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME'/records'
            #mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-AUC.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME'/AUC'
            #mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-CDF.png' $estimator'/'$gripper'/'$OBJ_SET_NAME'/AUC'
            #mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'.pth' $estimator'/'$gripper'/'$OBJ_SET_NAME'/pth'
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'.txt' $estimator'/'$gripper'/'$OBJ_SET_NAME'/bookmarks'

            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-loss.png' $estimator'/'$gripper'/'$OBJ_SET_NAME
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-trainingLoss.png' $estimator'/'$gripper'/'$OBJ_SET_NAME
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-accuracy.png' $estimator'/'$gripper'/'$OBJ_SET_NAME
            mv 'GraspSuccess-'$estimator'-'$OBJ_SET_NAME'-'$gripper'-'*'-lr.png' $estimator'/'$gripper'/'$OBJ_SET_NAME
        else
            echo '  Already trained '$estimator': '$OBJ_SET_NAME', '$gripper
            echo ''
        fi

        #########################################################  Report and analysis (requires all AUC files.)

        #python3 report_NObj1Grip.py -principal $estimator -objname $OBJ_SET_NAME -gripper $gripper -mode $MODE -epochs $EPOCHS -lr $LR -experiment $EXPERIMENT_NUMBER

        #mv $estimator'-'$OBJ_SET_NAME'-'$gripper'.md' $estimator'/'$gripper'/'$OBJ_SET_NAME
    done
done
