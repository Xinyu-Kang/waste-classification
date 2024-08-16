class SelectionStrategy:

    def __init__(self, image, segmentation_results, depth_results) -> None:
        self.image = image
        self.segmentation_results = segmentation_results
        self.depth_results = depth_results

    def select(self):
        depth_score_list = self.calculate_depth_score()
        sum_distance_list = self.sum_boarder_distance()
        min_distance_list = self.min_boarder_distance()
        area_list = self.calculate_area()
        image_grab_point = None
        label = None
        return image_grab_point, label

    def calculate_depth_score(self):
        pass

    def sum_boarder_distance(self):
        pass

    def min_boarder_distance(self):
        pass

    def calculate_area(self):
        pass