# Запуск

- Через Docker:

```
docker-compose up -d
```

- Консоль

Для установки необходимо пройти следующие шаги:

- Установка пакетов

```
pip install -r requirements.txt
```

- Запуск backend

```
uvicorn back:app --host 0.0.0.0 --port 8000
```
- Запуск frontend

```
http://localhost:8000/
```


# License
GPL-3.0 License

