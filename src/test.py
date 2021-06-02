import cv2
from Barberchop.ExtractionService import BarberchopYoloService

service = BarberchopYoloService('barberchop/weights/Barberchop_20210512.pt')
img = cv2.imread('barberchop/src/test.png')
results = service.extract(img)

print(f"Found {len(results)} barcodes")
for barcode in results:
    cv2.imshow("barcode", barcode.image)
    cv2.waitKey(1000)