import dynamixel_sdk as dxl         # Uses Dynamixel SDK library

"""
The servo class contains methods used to change attributes of the servo motors
most useful for setting up the servos to move when they are chained togerther
Beth Cutler
"""

class Servo(object):

    def __init__(self, portHandler, packetHandler, LED_colour, addresses, motor_id, device_name, torque_limit, speed_limit, max, min):
        
        self.port_handler = portHandler
        self.packet_handler = packetHandler

        self.LED_colour = LED_colour
        self.addresses = addresses
        self.motor_id = motor_id
        self.max = max
        self.min = min

        self.device_name = device_name
        self.torque_limit = torque_limit
        self.speed_limit = speed_limit


    def turn_on_LED(self):

        #ledColours = [0, 3, 2, 0, 7, 5, 0, 4, 6] may be useful if i refactor

        # write and read to servos
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["led"], self.LED_colour)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully LED turned on" %
                  self.motor_id)

    # limit the torque of the motors
    def limit_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["torque_limit"], self.torque_limit)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque limited" %
                  self.motor_id)
        
    # enable the torque of the motors
    def enable_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["torque_enable"], 1)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque enabled" %
                  self.motor_id)

    # disable the torque of the motors
    def disable_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["torque_enable"], 0)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully disabled torque" %
                  self.motor_id)

    # limit the speed of the servos
    def limit_speed(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["moving_speed"], self.speed_limit)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully speed limited" %
                  self.motor_id)

    #check whether the servo is moving using the inbuilt moving flag
    def moving_check(self):
        # write and read to servos
        dxl_moving_result = self.packet_handler.read1ByteTxRx(
            self.port_handler, self.motor_id, self.addresses["moving"])


        return int(dxl_moving_result[0])

    #if a a servo is given an angle that is outside of its range clip it to the max or min
    def verify_angle(self, angle): 
        
        if angle > self.max:
            angle = self.max
        elif angle < self.min:
            angle = self.min
        return angle

