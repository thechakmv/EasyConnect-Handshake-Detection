#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "handshake_model_data.cc"  // your model as C++ array

Adafruit_BNO055 bno = Adafruit_BNO055(55);

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kNumSamples = 50;   // 50 samples (e.g., 5s @ 10 Hz)
constexpr int kFeaturesPerSample = 6;

void setup() {
  Serial.begin(115200);
  delay(2000);

  if (!bno.begin()) {
    Serial.println("BNO055 not detected.");
    while (1);
  }

  model = tflite::GetModel(handshake_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();  // <-- change as needed for your model
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete. Starting in 2 seconds...");
  delay(2000);
}

void loop() {
  Serial.println("Collecting data for 5 seconds...");

  int sample_interval_ms = 100; // 10 Hz
  for (int i = 0; i < kNumSamples; ++i) {
    sensors_event_t accel, gyro;
    bno.getEvent(&accel, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bno.getEvent(&gyro, Adafruit_BNO055::VECTOR_GYROSCOPE);

    int base = i * kFeaturesPerSample;
    input->data.f[base + 0] = accel.acceleration.x;
    input->data.f[base + 1] = accel.acceleration.y;
    input->data.f[base + 2] = accel.acceleration.z;
    input->data.f[base + 3] = gyro.gyro.x;
    input->data.f[base + 4] = gyro.gyro.y;
    input->data.f[base + 5] = gyro.gyro.z;

    delay(sample_interval_ms);
  }

  Serial.println("Running inference...");

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
  } else {
    float result = output->data.f[0]; // assumes single float output
    Serial.print("Prediction score: ");
    Serial.println(result);

    if (result > 0.5) {
      Serial.println("🤝 Handshake detected!");
    } else {
      Serial.println("No handshake.");
    }
  }

  Serial.println("Waiting 10 seconds before next run...");
  delay(10000);
}
