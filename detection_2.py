import threading
from inference import InferencePipeline
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import uuid

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

tracker = sv.ByteTrack()

detection_polygon =np.array([[64, 1063],[3680, 1707],[3716, 2099],[84, 2075]]) # clear highway
detection_zone = sv.PolygonZone(polygon=detection_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT)) # highway
unique_cars = set()

filter_polygon = np.array([[44, 987],[3740, 1647],[3752, 2135],[32, 2119]]) # clear highway 
filter_zone = sv.PolygonZone(polygon=filter_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))

def get_crops(detections, image):

    result = {}
    for bbox, class_name in zip(detections.xyxy, detections.data["class_name"]):
        # print("class name in get crops: ", class_name)
        x1, y1, x2, y2 = bbox
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if class_name in result:
            result[class_name].append(crop)
        else:
            result[class_name] = [crop]

    return result

def background_task(results, frame):
    print("Background task")
    detections = sv.Detections.from_inference(results)
    # print("background detections", detections)
    car_features = detections[filter_zone.trigger(detections)]
    crops = get_crops(car_features, frame.image)
    # print("crops: ", crops)
    # his model has two classes. license-plate and vehicle
    # plate_img = Image.fromarray(crops["license plate"][0])
    # vehicle_img = Image.fromarray(crops["vehicle"][0])
    vehicle_img = Image.fromarray(crops["vehicle"][0])

    # total_width = plate_img.width + vehicle_img.width
    total_width = vehicle_img.width
    # max_height = max(plate_img.height, vehicle_img.height)
    max_height = vehicle_img.height

    combined_img = Image.new('RGB', (total_width, max_height))
    # combined_img.paste(plate_img, (0, 0))
    # combined_img.paste(vehicle_img, (plate_img.width, 0))
    combined_img.paste(vehicle_img)
    combined_img.show()

    # save the image with different names for each car
    unique_filename = f"./cropped_images/vehicle_{uuid.uuid1()}.png"
    try:
        combined_img.save(unique_filename)
        print("Image saved")
    except Exception as e:
        print("Error saving image: ", e)
def call_back(results, frame):
    """
    This function is called for each frame of the video.
    """
    # detections = sv.Detections.from_inference(results).with_nms(threshold=0.0)
    detections = sv.Detections.from_inference(results)
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(scene=frame.image.copy(), detections=detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)

    cv2.namedWindow("Vehicle Analytics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Analytics", 1600, 900)
    cv2.imshow("Vehicle Analytics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        pipeline.terminate()

    tracked_detections = tracker.update_with_detections(detections)
    # print("Tracked detections", tracked_detections)
    # print(f"Tracked detections:\nclass_id: {0}, class_name: {1}",tracked_detections.class_id, tracked_detections.data["class_name"])
    detected_cars = tracked_detections[tracked_detections.class_id == 0] # also change this according to class index
    # print("Detected cars", detected_cars)
    # print(f"Detected cars:\nclass_id: {0}, class_name: {1}", detected_cars.class_id, detected_cars.data["class_name"])
    cars_in_zone = detected_cars[detection_zone.trigger(detected_cars)]
    # print("Cars in zone", cars_in_zone)

    global unique_cars

    for car in cars_in_zone.tracker_id:
        print("Car ID: ", car)
        if not car in unique_cars:
            unique_cars.add(car)
            background_thread = threading.Thread(target=background_task, args=(results, frame))
            background_thread.start()

pipeline = InferencePipeline.init_with_yolo_world(
    video_reference=r".\videos\highway_2.mp4",
    classes=["vehicle"],
    model_size="m",
    confidence=0.05,
    on_prediction=call_back,
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
