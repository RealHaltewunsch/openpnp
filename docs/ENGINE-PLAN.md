# Gemeinsame Vision Engine für OpenPnP

## Ziel

**Ein lokaler Vision-Dienst lädt den gemeinsamen Bildencoder einmal pro Prozess. OpenPnP wählt die Aufgabe pro Anfrage, und der Dienst aktiviert den passenden trainierten Kopf.** Für IC-Unterseiten liefert der Kopf Mittelpunkt, Größe und Winkel. Für Homing liefert er Mittelpunkt und Sichtbarkeit. OpenPnP übernimmt die Kalibrierung, die Umrechnung von Bildpixeln in Maschinenkoordinaten und die Bewegungskorrektur.

Bei einem Aufgabenwechsel lädt der Dienst bei Bedarf nur den Kopf. Häufig genutzte Köpfe bleiben im Speicher. Neue Encoder-Gewichte erhalten eine eigene Modellversion. Jeder Kopf ist an eine kompatible Encoder-Version und deren Vorverarbeitung gebunden.

Der GitHub-Fork enthält seit `b3de186f` die Bottom-Vision-Schnittstelle `BottomVisionEngine`, den `LocalBottomVisionClient` und den Entwurf für einen externen Dienst. Baue auf diesen Schnittstellen auf und konkretisiere die Modellauswahl aus Abschnitt 3 in `docs/DINOv3-and-Packaging.md`. Das bisherige CNN-Homing dient beim späteren Wechsel als Vergleichsbasis. [Stand des Forks](https://github.com/RealHaltewunsch/openpnp/blob/b3de186f4e47c518e77951cee636b13644f22768/docs/DINOv3-and-Packaging.md)

## Plattformen und Reihenfolge

ONNX Runtime auf der CPU ist die gemeinsame Referenz für **Windows x64 und macOS arm64**. Prüfe DirectML auf der D700 und CoreML auf Apple Silicon mit denselben exportierten Graphen und Testbildern. So lässt sich je Plattform messen, welche Beschleunigung der Encoder und die Köpfe tatsächlich nutzen. [DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) · [CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

**Beginne mit IC-Unterseiten. Ergänze danach Homing als zweiten Kopf.** Weise mit beiden Aufgaben nach, dass der Dienst beim Wechsel dieselbe Encoder-Sitzung weiterverwendet.

## Direkter Arbeitsauftrag

Bearbeite die folgenden Schritte selbst im Fork. Beginne mit dem Schnittstellenvertrag und nutze ihn anschließend für die Implementierung. Halte pro Schritt kurz fest, was du geändert und wie du es geprüft hast.

1. **Vertrag entwerfen.** Einen versionierten Loopback-Dienst mit Health-Endpunkt und einer Inferenzroute definieren. Die Anfrage enthält Aufgabe, Bild und bei Bedarf erwartetes Gehäuse oder Suchbereich. Antworten tragen Status, Ergebnis in Originalpixeln und Modellversion. Bottom Vision liefert Mittelpunkt, Größe und Winkel einschließlich Mehrdeutigkeit. Homing liefert Mittelpunkt und Sichtbarkeit. Den vorhandenen Bottom-Client als Einstieg prüfen.
2. **Encoder und Köpfe trennen.** Mit eingefrorenem DINOv3 ViT-S/16 beginnen. Einen Encoder-Export und separate Head-Exporte erproben. Ein Manifest verknüpft Head-Version, Encoder-Version, Merkmalslayout und Vorverarbeitung. Der Dienst lädt den Encoder einmal, lädt Köpfe bei Bedarf und wärmt sie vor der ersten zeitkritischen Nutzung auf. Beim Aufgabenwechsel prüfen, dass die Encoder-Sitzung dieselbe bleibt. Einen Wechsel der Encoder-Gewichte als eigene Modellversion behandeln.
3. **Plattformen anhand gleicher Bilder prüfen.** ONNX Runtime CPU als Referenz auf Windows x64 und macOS arm64. Auf der Windows-D700 DirectML und auf Apple Silicon CoreML für genau die exportierten Graphen prüfen. Aktiven Provider und etwaige CPU-Ausführung protokollieren. Mittelpunkt- und Winkelabweichung, Speicher und Latenz messen. Die Übergabe der Encoder-Merkmale an den Kopf mitmessen. Start und Verbindung aus der vorhandenen Windows- und Mac-App testen. Für die erste Integration kann der Dienst separat laufen.
4. **Bottom Vision zuerst.** Einen Pose-Kopf auf echten IC-Unterseiten trainieren und auf unabhängigen Bildern gegen die aktuelle Pipeline prüfen. Mittelpunkt und Winkel auf den Footprint beziehen. Leere Düse, Fehlpick und symmetrische Gehäuse abdecken. Danach im Schattenmodus denselben Frame vergleichen. Erst nach Abnahme die bestehende OpenPnP-Korrektur mit dem neuen Ergebnis versorgen.
5. **Weitere Köpfe anschließen.** Homing als zweiten Kopf mit eigenen Labels und unabhängiger Abnahme ergänzen. Damit den Wechsel zwischen zwei Aufgaben mit demselben geladenen Encoder nachweisen. Platinenfiduziale, Feeder, Trays und Arbeitsflächenscan im Anschluss einzeln gemäß Roadmap migrieren.

Lies zum Einstieg Abschnitt 3 des bestehenden Entwurfs, die beiden Java-Klassen und den Aufruf in `ReferenceBottomVision`. Skizziere den Vertrag mit einer Beispielanfrage und je einer Antwort für Bottom Vision und Homing. Verwende diese Beispiele beim Aufbau des Dienstes als Testfälle.
