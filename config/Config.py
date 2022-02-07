class Config:
    class DataPaths:
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

        CropsFolder = "./data/crops/"
        ScaledDownCropsFolder = "./data/scaled_down_crops/"
        CropsFolder500Random = "./data/crops_500_random/"
        CropsFolder100Random = "./data/crops_100_random/"

        ClusterWheelsFolder = "./data/wheel_clusters/"

        UniqueRimsCollage = "./data/unique_rims_collage/"
        UniqueRimsAugmentation = "./data/unique_rims_augmentation/"
        WheelClassificationFolder = "./data/wheel_classification/"
        UniqueRimsCollageDataset = "./data/unique_rims_collage_dataset/"
        WheelClassificationAugmentation = "./data/wheel_classification_augmentation/"

        TrackingDatasetRandom = "./data/tracking_dataset/final/tracking_train/obj_train_data/"

        ScrewsSegmentationDatasetFolder = "./data/crops/"
    
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
