import threading
from inference import InferencePipeline
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import csv
import json


genai.configure(api_key="AIzaSyAh2OjBYiM5rsIc-rGiTUSQuBl-xAeYWpA")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

prompt_ = """Return analysis of the car. Include the color, make, type, and license plate. Return result in the following format like a python dictionary: {"color": "red", "make": "Toyota", "type": "car", "license plate": "ABC123"."""

def call_gemini(prompt, image):
    """
    Call the Gemini model with the given prompt and image.
    """
    global model
    response = model.generate_content([prompt, image])
    response.resolve()
    return response.text

report = r"D:\Internship II\vehicle_analytics\report_3.csv"
with open(report, mode="w", newline='') as f:
    fieldnames = ["Color", "Brand", "Car Type", "Plate"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

tracker = sv.ByteTrack()

detection_polygon = np.array([[130, 1756],[922, 1764],[954, 1888],[106, 1888]]) # big
detection_zone = sv.PolygonZone(polygon=detection_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT)) # highway
unique_cars = set()

filter_polygon = np.array([[106, 1720],[926, 1716],[982, 1900],[74, 1900]]) # clear highway 
filter_zone = sv.PolygonZone(polygon=filter_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))

cap = cv2.VideoCapture(r".\annotated_videos\highway_3.mp4")
ret, first_frame = cap.read()
frame_height, frame_width = first_frame.shape[:2]
# get fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
output_video_path = r".\videos\annotated_highway_3.mp4"  # Path for the output video
frame_rate = fps  # Adjust according to the input video's frame rate
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

def get_crops(detections, image):

    result = {}
    for bbox, class_name in zip(detections.xyxy, detections.data["class_name"]):
        print("class name in get crops: ", class_name)
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

    about = call_gemini(prompt_, vehicle_img)
    # brand = call_gemini("car make", vehicle_img)
    # car_type = call_gemini("car;van;suv;truck", vehicle_img)
    # plate = call_gemini("read the license plate", vehicle_img)

    # print(color, brand, car_type, plate)
    vehicle_data = json.loads(about)
    color = vehicle_data["color"]
    brand = vehicle_data["make"]
    car_type = vehicle_data["type"]
    plate = vehicle_data["license plate"]

    with open(report, mode="a", newline='') as f:
        fieldnames = ["Color", "Brand", "Car Type", "Plate"]
        data = [{"Color": color, "Brand":brand, "Car Type": car_type, "Plate": plate}]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(data)


    file_name = f"./vehicles/{color}-{brand}-{car_type}-{plate}.jpg"
    combined_img.save(file_name)

def call_back(results, frame):
    """
    This function is called for each frame of the video.
    """
    # detections = sv.Detections.from_inference(results).with_nms(threshold=0.0)
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
    # print("Tracked detections", tracked_detections)
    # print(f"Tracked detections:\nclass_id: {0}, class_name: {1}",tracked_detections.class_id, tracked_detections.data["class_name"])
    detected_cars = tracked_detections[tracked_detections.class_id == 0] # also change this according to class index
    # print("Detected cars", detected_cars)
    # print(f"Detected cars:\nclass_id: {0}, class_name: {1}", detected_cars.class_id, detected_cars.data["class_name"])
    cars_in_zone = detected_cars[detection_zone.trigger(detected_cars)]
    # print("Cars in zone", cars_in_zone)

    global unique_cars

    for car in cars_in_zone.tracker_id:
        # print("Car ID: ", car)
        if not car in unique_cars:
            unique_cars.add(car)
            background_thread = threading.Thread(target=background_task, args=(results, frame))
            background_thread.start()

pipeline = InferencePipeline.init_with_yolo_world(
    video_reference=r".\videos\highway_3.mp4",
    # classes=["white license plate of vehicle", "vehicle"],
    classes=["vehicle"],
    # classes=["license-plate", "car", "truck", "bus", "motorcycle"],
    model_size="m",
    confidence=0.05,
    on_prediction=call_back,
    
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
