import cv2
import pytesseract

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Read the image file
image = cv2.imread('car2.JPG')
cv2.imshow("Original", image)

# Convert to Grayscale Image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)

# Canny Edge Detection
canny_edge = cv2.Canny(gray_image, 170, 200)
cv2.imshow("Canny Edge", canny_edge)

# Find contours based on Edges
contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Initialize license plate contour and coordinates
contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

# Find the contour with 4 corners and create ROI
for contour in contours:
    # Find Perimeter of contour and ensure it is closed
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    print("Approx corners:", len(approx))
    if len(approx) == 4:  # Check if it is a rectangle
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = gray_image[y:y + h, x:x + w]
        break

# Apply thresholding
thresh, license_plate = cv2.threshold(license_plate, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Plate", license_plate)

# Remove Noise before OCR
license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)

# Text Recognition
text = pytesseract.image_to_string(license_plate)
print("License Plate:", text)

# Draw License Plate and write the text
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
image = cv2.putText(image, text, (x - 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Final Output", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
