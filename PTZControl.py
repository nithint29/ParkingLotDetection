from time import sleep
from onvif import ONVIFCamera

mycam = ONVIFCamera('192.168.204.111', 80, 'admin', 'admin')

# Create media service object
media = mycam.create_media_service()

# Create ptz service object
ptz = mycam.create_ptz_service()

# Get target profile
media_profile = media.GetProfiles()[0];

request = ptz.create_type('AbsoluteMove')
request.ProfileToken = media_profile._token

# time_space = 5
#
# for i in range(3):
#     # _x is for pan, _y is for tilt
#     request.Position.PanTilt = {'_x': -0.19, '_y': 0.7}
#     ptz.AbsoluteMove(request)
#     sleep(time_space)
#
#     request.Position.PanTilt = {'_x': -0.02, '_y': 0.82}
#     ptz.AbsoluteMove(request)
#     sleep(time_space)


    # request.Position.PanTilt = {'_x': 0.07, '_y': 0.85}
    # ptz.AbsoluteMove(request)
    # sleep(time_space)


request.Position.PanTilt = {'_x': -0.02, '_y': 0.82}
ptz.AbsoluteMove(request)
