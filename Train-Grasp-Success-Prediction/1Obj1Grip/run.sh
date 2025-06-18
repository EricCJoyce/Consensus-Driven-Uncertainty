ESTIMATORS='EPOS GDRNPP ZebraPose'
GRIPPERS='Parallel Underactuated'
BOP_OBJECTS='LMO1 LMO5 LMO6 LMO8 LMO9 LMO10 LMO11 LMO12 YCBV2 YCBV3 YCBV4 YCBV5 YCBV8 YCBV9 YCBV12'
#BOP_OBJECTS='LMO12 YCBV4 YCBV9 YCBV12'
EXPERIMENT_NUMBER=2                                                 #  However you named this folder in GitHub.
MODE='t3r3'                                                         #  30sep24: by decree, we are back on t3r3. Hope it doesn't suck now that we use cosine annealing.
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

        for bop_object in $BOP_OBJECTS                              #  For all objects...
        do
            if [ ! -d $estimator'/'$gripper'/'$bop_object ]         #  If no folder for this estimator, this gripper exists, create one.
            then
                mkdir $estimator'/'$gripper'/'$bop_object
            fi

            already_trained=true
            for dir in $estimator'/'$gripper'/'$bop_object'/pth' $estimator'/'$gripper'/'$bop_object'/AUC' $estimator'/'$gripper'/'$bop_object'/bookmarks' $estimator'/'$gripper'/'$bop_object'/confusion-matrices' $estimator'/'$gripper'/'$bop_object'/records'
            do
                if ! [ -d "$dir" ]
                then
                    already_trained=false
                    break
                fi
            done

            #########################################################  Train network, save and store related files.

            if [ "$already_trained" = false ]
            then
                echo 'TRAINING '$estimator': '$bop_object', '$gripper
                echo ''
                                                                    #  Train a network for this (principal, object, gripper).
                python3 train_1Obj1Grip.py -principal $estimator -object $bop_object -gripper $gripper -epochs $EPOCHS -optim Adam -lrsched $SCHED -lr $LR -lrmin $LR_MIN -batch 16 -mode $MODE -v -plot -report -save none

                mkdir $estimator'/'$gripper'/'$bop_object'/pth'
                mkdir $estimator'/'$gripper'/'$bop_object'/bookmarks'
                mkdir $estimator'/'$gripper'/'$bop_object'/records'
                mkdir $estimator'/'$gripper'/'$bop_object'/confusion-matrices'
                mkdir $estimator'/'$gripper'/'$bop_object'/AUC'
                                                                    #  Save the best records to the principal-gripper-object root.
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'$MODE'-best-confusionmatrix.txt' $estimator'/'$gripper'/'$bop_object
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'$MODE'-best-record.txt' $estimator'/'$gripper'/'$bop_object

                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'$MODE'-bestGeneralizing-confusionmatrix.txt' $estimator'/'$gripper'/'$bop_object
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'$MODE'-bestGeneralizing-record.txt' $estimator'/'$gripper'/'$bop_object

                                                                    #  Compute AUCs for every epoch.
                #for record in 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-record.txt'
                #do
                #    python3 auc_1Obj1Grip.py -nn $record -f avg -v
                #done
                                                                    #  Save everything else to their respective folders.
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-confusionmatrix.txt' $estimator'/'$gripper'/'$bop_object'/confusion-matrices'
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-record.txt' $estimator'/'$gripper'/'$bop_object'/records'
                #mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-AUC.txt' $estimator'/'$gripper'/'$bop_object'/AUC'
                #mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-CDF.png' $estimator'/'$gripper'/'$bop_object'/AUC'
                #mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'.pth' $estimator'/'$gripper'/'$bop_object'/pth'
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'.txt' $estimator'/'$gripper'/'$bop_object'/bookmarks'

                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-loss.png' $estimator'/'$gripper'/'$bop_object
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-trainingLoss.png' $estimator'/'$gripper'/'$bop_object
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-accuracy.png' $estimator'/'$gripper'/'$bop_object
                mv 'GraspSuccess-'$estimator'-'$bop_object'-'$gripper'-'*'-lr.png' $estimator'/'$gripper'/'$bop_object
            else
                echo '  Already trained '$estimator': '$bop_object', '$gripper
                echo ''
            fi

            #########################################################  Report and analysis (requires all AUC files.)

            #python3 report_1Obj1Grip.py -principal $estimator -object $bop_object -gripper $gripper -mode $MODE -epochs $EPOCHS -lr $LR -experiment $EXPERIMENT_NUMBER

            #mv $estimator'-'$bop_object'-'$gripper'.md' $estimator'/'$gripper'/'$bop_object
        done
    done
done
