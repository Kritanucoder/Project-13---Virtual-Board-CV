import cv2
import numpy as np
import mediapipe as mp

# --------------------------- Hand Detector --------------------------- #
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return fingers

        # Thumb (check if it's to the right of the previous joint)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check if they're above the second joint)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# --------------------------- Create Header with Color Options --------------------------- #
def createHeader(width=1280, height=125):
    header = np.zeros((height, width, 3), np.uint8)
    
    # Define colors and labels
    colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (50, 50, 50)]  # Pink, Blue, Green, Dark Gray(Eraser)
    labels = ["PINK", "BLUE", "GREEN", "ERASER"]
    
    regionWidth = width // len(colors)
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        x1 = i * regionWidth
        x2 = (i + 1) * regionWidth
        
        # Fill the region with color
        cv2.rectangle(header, (x1, 0), (x2, height), color, -1)
        
        # Add white border for better visibility
        cv2.rectangle(header, (x1, 0), (x2, height), (255, 255, 255), 3)
        
        # Add text label with contrasting color
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x1 + (regionWidth - text_size[0]) // 2
        text_y = height // 2 + text_size[1] // 2
        
        # Use white text for all colors for better visibility
        cv2.putText(header, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return header

# --------------------------- Main Script --------------------------- #
def main():
    brushThickness = 15
    eraserThickness = 50

    # Color options (updated eraser color to match header)
    colorOptions = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]  # Pink, Blue, Green, Black for eraser
    
    # Create header
    header = createHeader()
    drawColor = colorOptions[0]  # Start with pink
    
    # Region width for selection
    regionWidth = 1280 // len(colorOptions)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handDetector(detectionCon=0.75, maxHands=1)
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    print("Controls:")
    print("- Hold up INDEX and MIDDLE fingers to select colors")
    print("- Hold up only INDEX finger to draw")
    print("- Press 'q' to quit")
    print("- Press 'c' to clear canvas")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Find hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) >= 21:  # Ensure all landmarks are detected
            # Get finger tip positions
            x1, y1 = lmList[8][1:]   # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip
            
            # Check which fingers are up
            fingers = detector.fingersUp()

            # Selection Mode: index and middle fingers up
            if fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0
                # Check if fingers are in the header region
                if y1 < 125:
                    # Determine which color region
                    idx = min(x1 // regionWidth, len(colorOptions) - 1)
                    drawColor = colorOptions[idx]
                    print(f"Selected color: {['Pink', 'Blue', 'Green', 'Eraser'][idx]}")
                
                # Draw selection rectangle
                cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

            # Drawing Mode: only index finger up
            elif fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Choose thickness based on color (eraser vs brush)
                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                
                # Draw on both the live feed and canvas
                cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                
                xp, yp = x1, y1

        # Merge canvas with live feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Add header to the image
        img[0:125, 0:1280] = header

        # Create top-bottom display layout with TRUE 50-50 split
        # Use smaller, more manageable dimensions for true 50-50 split
        total_width = 1280
        total_height = 800  # Smaller total height for better 50-50 visibility
        
        # Calculate exact 50-50 dimensions
        camera_height = total_height // 2  # Exactly 400px (50%)
        canvas_height = total_height // 2  # Exactly 400px (50%)
        display_width = total_width
        
        # Resize images to exact 50-50 dimensions
        img_resized = cv2.resize(img, (display_width, camera_height))
        canvas_resized = cv2.resize(imgCanvas, (display_width, canvas_height))
        
        # Create combined display (vertical stack) - this ensures exact 50-50
        combined = np.vstack((img_resized, canvas_resized))
        
        # Add labels with appropriate positioning
        font_scale = 0.8
        label_thickness = 2
        
        # Live camera label (top section)
        cv2.putText(combined, "LIVE CAMERA", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), label_thickness)
        
        # Canvas label (bottom section) - positioned exactly at 50% mark
        cv2.putText(combined, "CANVAS", (20, camera_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), label_thickness)
        
        # Display the combined result
        cv2.imshow("Virtual Drawing Board - Camera | Canvas", combined)
        
        # Position window to fill screen
        cv2.moveWindow("Virtual Drawing Board - Camera | Canvas", 0, 0)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            print("Canvas cleared!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()