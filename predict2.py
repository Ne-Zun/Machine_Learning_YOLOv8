import cv2
from ultralytics import YOLO

# Load the trained model
# Replace this path with the correct path to your 'best.pt' file
model = YOLO(r"F:/MACHINE_TRAFFIC/runs/detect/train3/weights/best.pt")

# Set the path to your input video
input_video_path = "F:/MACHINE_TRAFFIC/vdeo2.mp4"

# Set the path for the output video
output_video_path = "/MACHINE_TRAFFIC/results_video8.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties to create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Processing video...")
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was read successfully
        if ret:
            # Run the YOLO model on the current frame
            # The 'stream=True' argument is more efficient for video processing
            results = model(frame, stream=True)

            # Loop through the results to get the annotated frame
            for r in results:
                # Get the annotated image (frame with bounding boxes)
                im_array = r.plot()

                # Convert the array to BGR format for OpenCV
                annotated_frame = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

                # Write the annotated frame to the output video
                out.write(annotated_frame)

            # Optional: Display the frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # End of video
            break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing finished. Results saved to: {output_video_path}")



