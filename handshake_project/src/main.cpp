#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "handshake_model.cc"  // TFLite model array

Adafruit_BNO055 bno = Adafruit_BNO055(55);

constexpr int kTensorArenaSize = 15 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

constexpr int kInferenceIntervalMs = 1000;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kNumSamples = 101;
constexpr int kFeaturesPerSample = 6;
constexpr int kInputSize = kNumSamples * kFeaturesPerSample;
constexpr int kSampleIntervalMs = 10;  // 100 Hz

float circular_buffer[kInputSize] = {0};  // rolling window buffer
int current_sample = 0;
bool buffer_filled = false;

void setup() {
  Serial.begin(115200);
  delay(2000);

  if (!bno.begin()) {
    Serial.println("BNO055 not detected.");
    while (1);
  }

  model = tflite::GetModel(handshake_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();
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

  Serial.println("Setup complete. Listening for handshakes...");
}

void loop() {
  static unsigned long last_sample_time = 0;
  if (millis() - last_sample_time < kSampleIntervalMs) return;
  last_sample_time = millis();

  // Collect IMU data
  sensors_event_t accel, gyro;
  bno.getEvent(&accel, Adafruit_BNO055::VECTOR_LINEARACCEL);
  bno.getEvent(&gyro, Adafruit_BNO055::VECTOR_GYROSCOPE);

  // Circular buffer insert
  int base = (current_sample % kNumSamples) * kFeaturesPerSample;
  circular_buffer[base + 0] = accel.acceleration.x;
  circular_buffer[base + 1] = accel.acceleration.y;
  circular_buffer[base + 2] = accel.acceleration.z;
  circular_buffer[base + 3] = gyro.gyro.x;
  circular_buffer[base + 4] = gyro.gyro.y;
  circular_buffer[base + 5] = gyro.gyro.z;

  current_sample++;
  if (current_sample >= kNumSamples) buffer_filled = true;

  if (!buffer_filled) return;  // wait until buffer is full

  // Copy circular buffer to model input in linear order
  for (int i = 0; i < kNumSamples; ++i) {
    int circular_index = ((current_sample + i) % kNumSamples) * kFeaturesPerSample;
    int linear_index = i * kFeaturesPerSample;
    for (int j = 0; j < kFeaturesPerSample; ++j) {
      input->data.f[linear_index + j] = circular_buffer[circular_index + j];
    }
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  float result = output->data.f[0];
  Serial.print("Score: ");
  Serial.print(result, 3);
  Serial.print(" — ");
  Serial.println(result > 0.5 ? "Handshake Detected" : "No Handshake");
}
