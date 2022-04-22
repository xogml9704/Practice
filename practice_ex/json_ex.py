# json 파일 읽기
import json

with open('D:\\code\\data\\json_test\\test.json', 'r') as f:
    json_data = json.load(f)
print(json.dumps(json_data))

# json 파일 깔끔하게 출력
print(json.dumps(json_data, indent="\t"))

# json 파일 출력하기
k5_price = json_data['K5']['price']
print(k5_price)

# json 파일 수정하기
# 값을 불러오는 것 뿐만 아니라 쓰는 것도 가능
json_data['K5']['price'] = "7000"
print(json_data['K5']['price'])

with open('D:\\code\\data\\json_test\\test.json', 'w', encoding='utf-8') as make_file:

    json.dump(json_data, make_file, indent="\t")

	

with open('D:\\code\\data\\json_test\\test.json', 'r') as f:

    json_data = json.load(f)

print(json.dumps(json_data, indent="\t"))