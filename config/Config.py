class Config:
    class DataPaths:
        DataFolder = "./data/"
        ZippedRawDataFolder = "./data/raw/"
        VideoFolder = "./data/video/"
        FramesFolder = "./data/frames/"

        VideoFolderExample = "./data/video/day/"
        FrameFolderExample = "./data/frames/day/time/camera/"

        VideoFileExample = "./data/video/day/day_time_camera.mp4"
        FrameFileExample = "./data/frames/day/time/camera/framexxx.jpg"

        RandomFrameSampleFolder = "./data/random_sample_frames/"

        TrackingYoloModel = "./model/tracking_yolo.pt"
        WheelBoltsDetectionYoloModel = "./model/whee_bolts_detection_yolo.pt"
        SizeEstimation256Model = "./model/size_estimation_256_yolo.pt"
        TrackingV2YoloModel = "./model/tracking_v2_yolo.pt"

        CropsFolder = "./data/crops/"
        ScaledDownCropsFolder = "./data/downscaled_crops/"
        CropsFolder500Random = "./data/crops_500_random/"
        CropsFolder100Random = "./data/crops_100_random/"

        ClusterWheelsFolder = "./data/wheel_clusters/"

        UniqueRimsCollage = "./data/unique_rims_collage/"
        UniqueRimsAugmentation = "./data/unique_rims_augmentation/"
        WheelClassificationFolder = "./data/wheel_classification/"
        UniqueRimsCollageDataset = "./data/unique_rims_collage_dataset/"
        UniqueRimsCollageDatasetClipped = "./data/unique_rims_collage_dataset_clipped/"
        UniqueRimsCollageDatasetClippedGeometryOnly = "./data/unique_rims_collage_dataset_clipped_geometry_only/"
        WheelClassificationAugmentation = "./data/wheel_classification_augmentation/"
        AugmentationOfClasses = "./data/augmentation_examples/"

        Dataset31NotSelected = "./data/dataset31_not_selected/"
        Dataset31 = "./data/dataset31/"

        WheelAndBoltsDatasetSamples = "./data/wheel_and_bolts_dataset_samples/"

        SizeEstimationDatasetTrain = "./data/size_estimation_dataset/size_estimation_dataset_train/obj_train_data/"
        SizeEstimationDatasetTest = "./data/size_estimation_dataset/size_estimation_dataset_test/obj_train_data/"
        SizeEstimationDatasetVal = "./data/size_estimation_dataset/size_estimation_dataset_val/obj_train_data/"


        TrackingDatasetRandom = "./data/tracking_dataset/final/tracking_train/obj_train_data/"
        TrackingDatasetRandomValidation = "./data/tracking_dataset/final/tracking_validation/obj_train_data/"
        TrackingDatasetRandomTest = "./data/tracking_dataset/final/tracking_test/obj_train_data/"

        TrackingDatasetV2RandomTrain = "./data/tracking_dataset_v2/tracking_dataset_train/obj_train_data/" 
        TrackingDatasetV2RandomValidation = "./data/tracking_dataset_v2/tracking_dataset_val/obj_train_data/"
        TrackingDatasetV2RandomTest = "./data/tracking_dataset_v2/tracking_dataset_test/obj_train_data/"

        ScrewsSegmentationDatasetFolder = "./data/crops/"

        ImageGridsGeneratedFolder = "./data/image_grids_generated/"

        
    
    class Tracking:
        HoughTuningDownscaleValue = 40
        HoughTuningBlurValue = 5
        HoughTuningCanny1Value = 150 
        HoughTuningCanny2Value = 220
        HoughTuningDpValue = 1
        HoughTuningMinDistValue = 120
        HoughTuningParam1Value = 205
        HoughTuningParam2Value = 50
        HoughTuningMinRadiusValue = 70
        HoughTuningMaxRadiusValue = 130
