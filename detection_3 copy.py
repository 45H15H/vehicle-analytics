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

detection_polygon = np.array([[130, 1756],[922, 1764],[954, 1888],[106, 1888]]) # Slightly smaller than the filter zone
detection_zone = sv.PolygonZone(polygon=detection_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))
unique_cars = set()

filter_polygon = np.array([[106, 1720],[926, 1716],[982, 1900],[74, 1900]]) # Slightly larger than the detection zone
filter_zone = sv.PolygonZone(polygon=filter_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))

# To get the frame size of the video
cap = cv2.VideoCapture(r".\videos\highway_3.mp4")
ret, first_frame = cap.read()
frame_height, frame_width = first_frame.shape[:2]
# get fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
output_video_path = r".\annotated_videos\annotated_highway_3.mp4"  # Path for the output video
frame_rate = fps  # Adjust according to the input video's frame rate
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

def get_crops(detections, image):
    result = {}
    for bbox, class_name in zip(detections.xyxy, detections.data["class_name"]):
        x1, y1, x2, y2 = bbox
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if class_name in result:
            result[class_name].append(crop)
        else:
            result[class_name] = [crop]

    return result

def background_task(results, frame):
    detections = sv.Detections.from_inference(results)
    car_features = detections[filter_zone.trigger(detections)]
    crops = get_crops(car_features, frame.image)
    vehicle_img = Image.fromarray(crops["vehicle"][0])

    total_width = vehicle_img.width
    max_height = vehicle_img.height

    combined_img = Image.new('RGB', (total_width, max_height))
    combined_img.paste(vehicle_img)
    combined_img.show() # comment this line if you don't want to see the image

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
    detections = sv.Detections.from_inference(results) 
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(scene=frame.image.copy(), detections=detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)

    cv2.namedWindow("Vehicle Analytics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Analytics", 506, 900)
    cv2.imshow("Vehicle Analytics", annotated_frame)
    video_writer.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        pipeline.terminate()

    tracked_detections = tracker.update_with_detections(detections)
    detected_cars = tracked_detections[tracked_detections.class_id == 0] # also change this according to class index
    cars_in_zone = detected_cars[detection_zone.trigger(detected_cars)]

    global unique_cars

    for car in cars_in_zone.tracker_id:
        print("Car ID: ", car)
        if not car in unique_cars:
            unique_cars.add(car)
            background_thread = threading.Thread(target=background_task, args=(results, frame))
            background_thread.start()

pipeline = InferencePipeline.init_with_yolo_world(
    video_reference=r".\videos\highway_3.mp4",
    classes=["vehicle"],
    model_size="l",
    confidence=0.03,
    on_prediction=call_back,
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
