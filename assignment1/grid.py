import exifread

origin_file_path = './check7.jpg'

with open(origin_file_path, "rb") as image_file:
    tags = exifread.process_file(image_file)

for key in tags.keys():
    value = str(tags[key])
    print('{0}:{1}'.format(key, value))