#include <Servo.h>

Servo trg;
Servo pit;
Servo yaw;
unsigned long offset = 0;  // cameraTime - arduinoTime

void setup() {
  Serial.begin(115200);
  trg.attach(11);
  pit.attach(10);
  yaw.attach(9);

  trg.write(110);
  pit.write(90);
  yaw.write(90);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd.startsWith("SYNC")) {
      // SYNC <cam_t> <delay>
      int s1 = cmd.indexOf(' ');
      int s2 = cmd.indexOf(' ', s1 + 1);
      unsigned long cam_t = strtoul(cmd.substring(s1 + 1, s2).c_str(), NULL, 10);
      unsigned long delay = strtoul(cmd.substring(s2 + 1).c_str(), NULL, 10);
      unsigned long ar_t  = micros();
      offset = (cam_t + delay) - ar_t;
    }
    else if (cmd.startsWith("FIRE")) {
      // FIRE <pitch> <yaw> <cam_time>
      int firstSpace = cmd.indexOf(' ');
      int secondSpace = cmd.indexOf(' ', firstSpace + 1);
      int thirdSpace = cmd.indexOf(' ', secondSpace + 1);

      int pitch_angle = cmd.substring(firstSpace + 1, secondSpace).toInt();
      int yaw_angle   = cmd.substring(secondSpace + 1, thirdSpace).toInt();
      unsigned long T_cam = strtoul(cmd.substring(thirdSpace + 1).c_str(), NULL, 10);

      // Convert camera time → Arduino time
      unsigned long T_arduino = T_cam - offset;

      // Move servos first
      pit.write(pitch_angle);
      yaw.write(yaw_angle);

      while ((long)(micros() - T_arduino) < 0) {
        // busy wait for highest precision
      }

      trg.write(60);
      delayMicroseconds(1000000); // adjust as needed
      trg.write(110);
    }
    else if (cmd.startsWith("trigger ")) {
      int v = cmd.substring(8).toInt();
      trg.write(v);
    } 
    else if (cmd.startsWith("yaw ")) {
      int v = cmd.substring(4).toInt();
      yaw.write(v);
    }
    else if (cmd.startsWith("pitch ")) {
      int v = cmd.substring(6).toInt();
      pit.write(v);
    }
  }
}
