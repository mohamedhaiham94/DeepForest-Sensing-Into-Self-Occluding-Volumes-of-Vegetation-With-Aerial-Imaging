from controllers import MultiSpectralProcessor

from modules import CalibrationStep
from modules import CropBordersStep
from modules import LensCorrectionStep
from modules import ImageAlignmentStep
from modules import SaveGrayscaleStep


if __name__ == "__main__":

    # Example image data and reference band
    dataset_path = 'dataset'
    reference_band = "NIR"
  
    # Create the processing controller with configuration  
    controller = MultiSpectralProcessor(dataset_path, 
                                        CalibrationStep, 
                                        SaveGrayscaleStep,
                                        LensCorrectionStep, 
                                        ImageAlignmentStep, 
                                        CropBordersStep
                                    )

    # Process the images
    final_images = controller.process()
