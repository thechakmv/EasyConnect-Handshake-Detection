#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);



void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  if (!bno.begin()) {
    Serial.println("Failed to detect BNO055. Check connections.");
    while (1);
  }
  else {
    Serial.println("Detected BNO055.");
    bno.setExtCrystalUse(true);
  }
}

void loop() {

  //Start up the sensor
  sensors_event_t event;
  bno.getEvent(&event);

  //Orientation
  Serial.print(event.orientation.x); Serial.print(", "); //Heading
  Serial.print(event.orientation.y); Serial.print(", "); //Roll
  Serial.print(event.orientation.z); Serial.print(", "); //Pitch
  
  //Accelerometer
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

  Serial.print(accel.x()); Serial.print(", "); //X
  Serial.print(accel.y()); Serial.print(", "); //Y
  Serial.print(accel.z()); Serial.print(", "); //Z

  //GyroScope
  imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  Serial.print(gyro.x()); Serial.print(", "); //X
  Serial.print(gyro.y()); Serial.print(", "); //Y
  Serial.println(gyro.z());//Z

  delay(100);
}
