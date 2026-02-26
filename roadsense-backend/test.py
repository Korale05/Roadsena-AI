# test.py — put this in your roadsense-backend folder and run: python test.py
import requests

url   = "http://localhost:5000/api/upload-image"
files = {"file": open(r"D:\Pathhole_Detection\path1.jpg", "rb")}
data  = {"road_name": "Test Road MG Solapur", "lat": "17.6868", "lng": "75.9091"}

print("Sending image to AI...")
response = requests.post(url, files=files, data=data)
result   = response.json()

if result["success"]:
    print("\n✅ SUCCESS!")
    print(f"Defects Found  : {result['data']['score']['defect_count']}")
    print(f"Severity Score : {result['data']['score']['severity_score']}/10")
    print(f"Road Quality   : {result['data']['score']['road_quality_index']}/100")
    print(f"Status         : {result['data']['score']['severity_level']}")
    print(f"Urgency        : {result['data']['score']['repair_urgency']}")
    print(f"Economic Impact: Rs.{result['data']['score']['economic_impact_rs']:,}")
    print(f"Annotated Image: http://localhost:5000/outputs/{result['data']['annotated_image']}")
else:
    print(f"\n❌ ERROR: {result['error']}")