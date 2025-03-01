import os
import shutil

class DatasetHandler:

    def __init__(self, images_path: str, added_new_images: bool = False):

        self.images_path = images_path
        
        images: list[str] = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith(".")]
        
        if added_new_images:
            for num, image in enumerate(sorted(images), start = 1):
                os.rename(image, f"{num}.jpg") # Rinomino le immagini con numeri crescenti
                shutil.move(os.path.join(os.getcwd(), f"{num}.jpg"), images_path) # Riporto le immagini rinominate all'interno della corretta directory
                images: list[str] = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith(".")]
                
        self.__dataset: list[str] = [image for image in sorted(images)]

    def get_dataset_list(self) -> list[str]:
        return self.__dataset
    
    def len_dataset(self) -> int:
        return len(self.__dataset)    
    
    def get_image_path_from_image_name(self, name: str) -> str:
        return os.path.join(self.images_path, f"{name}.jpg")
    
    def remove_image_from_list_by_name(self, name: str) -> None:
        self.__dataset.remove(os.path.join(self.images_path, f"{name}.jpg"))