ChatExport_2025-04-05 - стандартная выгрузка сообщений телеграм-канала https://t.me/mod_russia в формате json (result.json)
RMDR_LabelStudio_WebService - web service, который использует обученную модель для возвращения анотаций(предсказаний) в Label Studio. Web Service можно использовать для дальнейшей разметки данных или для просмотра результатов работы модели.
web service реализован в виде докера, для запуска котрого нужно использовать docker-compose.yml  (docker compose up). Исходный код web service находится в файле model.py. Обученная модель (веса) находится по пути RMDR_LabelStudio_WebService/rmdr_ner_docker/data/server/models/rmdr_ner_model.bin
NLP2025_RMDR_GetData.ipynb - Отчистка данных. Удаляет лишние символы и сохраняет результат в атрибут clearText объекта message. Результат сохраятся в файл RMDR.json
NLP_2025_RMDR_NERLM_2.ipynb - Алгоритм обучения модели. Обученная модель сохранятся в файл results/rmdr_ner_model.bin
RMDR_ANATATION_3_MONTH.json - Данные для обучения с разметкой выполненой в Label Studio. В файле размечено только 3 месяца сводок: июнь-июль 2022, июнь 2023, июнь 2024. 
