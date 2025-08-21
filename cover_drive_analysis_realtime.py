import cv2
import mediapipe as mp
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Input and Output video paths
    video_path = 'input_video.mp4'
    output_path = 'output/annotated_video.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # FPS calculation
    prev_time = 0
    curr_time = 0

    # Data for temporal analysis
    elbow_angles = []
    spine_leans = []

    # Evaluation data
    evaluation = {
        "footwork": [],
        "head_position": [],
        "swing_control": [],
        "balance": [],
        "follow_through": []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        curr_time = time.time()
        fps_calc = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps_calc)}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and find pose
        results = pose.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for all required landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            # 1. Front elbow angle
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            elbow_angles.append(elbow_angle)
            cv2.putText(image, f"Elbow: {int(elbow_angle)} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if elbow_angle > config['elbow_angle_threshold']:
                cv2.putText(image, "Good elbow elevation", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                evaluation["swing_control"].append(1)
            else:
                evaluation["swing_control"].append(0)

            # 2. Spine lean
            spine_lean = calculate_angle(left_hip, left_shoulder, (left_shoulder[0], left_shoulder[1] - 1))
            spine_leans.append(spine_lean)
            cv2.putText(image, f"Spine Lean: {int(spine_lean)} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 3. Head-over-knee vertical alignment
            head_knee_dist = abs(nose[0] - left_knee[0]) * frame_width
            cv2.putText(image, f"Head-Knee Align: {int(head_knee_dist)} px", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if head_knee_dist < config['head_knee_distance_threshold']:
                cv2.putText(image, "Head over front knee", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                evaluation["head_position"].append(1)
            else:
                cv2.putText(image, "Head not over front knee", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                evaluation["head_position"].append(0)

            # 4. Front foot direction
            foot_direction = calculate_angle(left_knee, left_ankle, left_foot_index)
            cv2.putText(image, f"Foot Angle: {int(foot_direction)} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if foot_direction > config['foot_angle_lower_threshold'] and foot_direction < config['foot_angle_upper_threshold']:
                evaluation["footwork"].append(1)
            else:
                evaluation["footwork"].append(0)

            # 5. Balance (via spine lean)
            if spine_lean < config['spine_lean_threshold']:
                evaluation["balance"].append(1)
            else:
                evaluation["balance"].append(0)

            # 6. Follow-through
            # Simple check: after max elbow extension, does the elbow stay high?
            if len(elbow_angles) > 10: # Ensure we have some frames to analyze
                max_extension_frame = np.argmin(elbow_angles)
                # Check frames after max extension
                if len(elbow_angles) - max_extension_frame > 5: # Ensure there are frames after extension
                    follow_through_angles = elbow_angles[max_extension_frame:]
                    avg_follow_through_angle = np.mean(follow_through_angles)
                    if avg_follow_through_angle > config['elbow_angle_threshold']:
                        evaluation["follow_through"].append(1)
                    else:
                        evaluation["follow_through"].append(0)
                else:
                    evaluation["follow_through"].append(0) # Not enough frames for follow through
            else:
                evaluation["follow_through"].append(0) # Not enough frames to analyze


        except Exception as e:
            # print(f"Error processing frame: {e}")
            pass

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Write the frame to the output video
        out.write(image)

        # Display the resulting frame
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Final Evaluation
    final_scores = {}
    for category, values in evaluation.items():
        if values:
            score = np.mean(values) * 10
            final_scores[category] = {"score": score, "feedback": ""}
        else:
            final_scores[category] = {"score": 0, "feedback": "No data"}

    # Add feedback based on scores (example)
    if final_scores["swing_control"]["score"] > 7:
        final_scores["swing_control"]["feedback"] = "Excellent swing control."
    else:
        final_scores["swing_control"]["feedback"] = "Work on maintaining a high elbow."

    if final_scores["head_position"]["score"] > 7:
        final_scores["head_position"]["feedback"] = "Good head position over the front knee."
    else:
        final_scores["head_position"]["feedback"] = "Try to keep your head over your front knee."

    if final_scores["footwork"]["score"] > 7:
        final_scores["footwork"]["feedback"] = "Excellent footwork, pointing towards the shot."
    else:
        final_scores["footwork"]["feedback"] = "Work on your front foot placement."

    if final_scores["balance"]["score"] > 7:
        final_scores["balance"]["feedback"] = "Good balance throughout the shot."
    else:
        final_scores["balance"]["feedback"] = "Try to maintain a more stable core."

    if final_scores["follow_through"]["score"] > 7:
        final_scores["follow_through"]["feedback"] = "Excellent follow-through, with a high elbow."
    else:
        final_scores["follow_through"]["feedback"] = "Work on completing your shot with a full follow-through."

    # Skill Grade Prediction
    average_score = np.mean([cat["score"] for cat in final_scores.values() if cat["score"] > 0])
    if average_score >= 8:
        skill_grade = "Advanced"
    elif average_score >= 5:
        skill_grade = "Intermediate"
    else:
        skill_grade = "Beginner"
    final_scores["skill_grade"] = skill_grade

    # Save evaluation to a file
    evaluation_json_path = 'output/evaluation.json'
    evaluation_report_path = 'output/evaluation_report.txt'
    temporal_smoothness_path = 'output/temporal_smoothness.png'

    with open(evaluation_json_path, 'w') as f:
        json.dump(final_scores, f, indent=4)

    # Create and save temporal smoothness plot
    plt.figure(figsize=(10, 5))
    plt.plot(elbow_angles, label='Elbow Angle')
    plt.plot(spine_leans, label='Spine Lean')
    plt.title('Temporal Smoothness')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.savefig(temporal_smoothness_path)

    # Generate and save text report
    with open(evaluation_report_path, 'w') as f:
        f.write("AthleteRise - AI-Powered Cricket Analytics Report\n")
        f.write("====================================================\n\n")
        f.write(f"Overall Skill Grade: {final_scores['skill_grade']}\n\n")
        for category, data in final_scores.items():
            if category != 'skill_grade':
                f.write(f"{category.replace('_', ' ').title()}:\n")
                f.write(f"  Score: {data['score']:.2f}/10\n")
                f.write(f"  Feedback: {data['feedback']}\n\n")

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()