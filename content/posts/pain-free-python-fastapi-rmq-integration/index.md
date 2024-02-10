---
title: 'Pain-free Python Fastapi RabbitMQ Integration'
date: 2024-02-10T06:55:09+07:00
tags: ["event-driven", "messagebroker", "python"]
draft: false
description: "Despite of the powerfulness of FastAPI, I found it's not easy to work with threads and RabbitMQ. I was struggled for 3 days to find the solution. And here I'll share my findings on creating both RMQ based producer and consumer in single FastAPI app."
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
cover:
    image: "cover.jpeg" # image path/url
    alt: "Three pikachus delivering packages" # alt text
    caption: "This image was generated using Bing Image Creator by Microsoft" # display caption under cover
    relative: true # when using page bundles set this to true
math: katex
keywords: ["messagebroker", "rabbitmq", "fastapi", "python"]
summary: "Despite of the powerfulness of FastAPI, I found it's not easy to work with threads and RabbitMQ. Here I'll share my findings on creating both RMQ based producer and consumer in single FastAPI app."
---


## Background
I was working on implementing RabbitMQ (RMQ) in FastAPI for doing background task. Initially, there was a long process in our system, and the execution time is not deterministic. Making the some request caught in the timeout error. To solve it, we come up with a message broker based solution.

In this case, our system has to communicate with itself to do the long process in the background. So the origin request won't be expired. Optionally, once the process is done it can be asked to notify the user.

![Illustration of the system](images/system.png#center)
The illustration of our FastAPI interaction with RabbitMQ producer and consumer.


## Pub/Sub Pattern
The publisher/subscriber pattern using message broker is one of the common pattern. It also covered in the [official documentation of RabbitMQ](https://www.rabbitmq.com/tutorials/tutorial-three-python.html). However, the common pattern is to have publisher and consumer as different application because usually the communication is happened between two or more services. While in our case, we don't want to create another service just for processing our task in background. Therefore, we need to have both publisher and subscriber in a single service. 

Usually, in more complex system there could be multiple publisher and subscriber for different events. But in this case, we will demonstrate it using only one publisher and subscriber which we will call it as producer and consumer.
![pub/sub illustration](images/pubsub.png#center)
The simple illustration of single pub/sub.


## Integration with FastAPI
Initially, I did my experiments by running producer and consumer in dedicated threads; and using asyncio connection. The threads solution always gave me an error `pika.exceptions.StreamLostError: Stream connection lost: BrokenPipeError(32, 'Broken pipe')` after idle for several minutes. While the asyncio based one always blocking the same thread that used to serve endpoint routes. Meaning when the background process is running, any calls to the endpoints should be wait for it to be finished.

Finally, I found the solution using [`aio-pika`](https://aio-pika.readthedocs.io/en/latest/) library, and the decided to built my solution on top of it. So, let's start demonstrate it using simple FastAPI project.

> *Note: I won't explain basic terms and definitions of each RMQ part. Feel free to visit their official tutorial to better understand the terms and definitions*
>
> [RabbitMQ (Pika) documentation](https://www.rabbitmq.com/tutorials/tutorial-one-python.html)
>
> [`aio-pika` documentation](https://aio-pika.readthedocs.io/en/latest/)

First thing first, lets create a `rmq.py` file for our `PikaClient` which will be interact with RMQ.
```python3
import logging
import aio_pika
import asyncio


logger = logging.getLogger(__name__)


class PikaClient():

    def __init__(self, queue_name: str, exchange_name: str, conn_str: str) -> None:
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.conn_str = conn_str

        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None


    async def start_connection(self):
        logger.info("Starting a new connection")
        self.connection = await aio_pika.connect_robust(url=self.conn_str)

        logger.info("Opening a new channel")
        self.channel = await self.connection.channel()

        logger.info("Declaring an exchange: %s" % self.exchange_name)
        self.exchange = await self.channel.declare_exchange(name=self.exchange_name, type=aio_pika.ExchangeType.DIRECT)

        await self.setup_queue()


    async def setup_queue(self):
        logger.info("Setup a queue: %s" % self.queue_name)
        self.queue = await self.channel.declare_queue(name=self.queue_name)

        logger.info("Bind queue to exchange")
        await self.queue.bind(self.exchange)


    async def disconnect(self):
        try:
            if not self.connection.is_closed:
                await self.connection.close()
        except Exception as _e:
            logger.error(_e)
```
Here we just created the basic interface for interacting with RMQ. It consists of starting a new RMQ connection, opening a new channel, declaring an exchange, and setup a queue.

Then, let's add the following functions to enable run `PikaClient` as producer.
```python3
    async def start_producer(self):
        await self.start_connection()
        logger.info("Producer has been started")

        return self
        

    async def publish_message(self, message):
        await self.exchange.publish(
            aio_pika.Message(body=message.encode()),
            routing_key=self.queue_name
        )
```
Once the client is connected and ran in producer mode, it can call `publish_message()` at any time to send the message to RMQ.

Now, let's continue add the following functions for enabling run as consumer.
```python3
    async def start_consumer(self):
        await self.start_connection()

        await self.channel.set_qos(prefetch_count=1)

        logger.info("Starting consumer")
        await self.queue.consume(self.handle_message)

        logger.info("Consumer has been started")

        return self
    

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage):

        # simulating long process
        await asyncio.sleep(10)

        logger.info("Consumer: Got message from producer: %s" % message.body.decode())

        await message.ack()
```
Here, we need to set few things up at the initial connection. 

The `channel.set_qos()` define how many job allowed to be executed concurrently. The `queue.consume(callback)` trigger the consumer to subscribe/listen to the predefined queue, waiting for any new message. 

The callback function (in this case `handle_message()`) is executed rightaway the message arrived in RMQ. Note that we simulating the long background process using `asyncio.sleep(10)`. Later you will see that during long process simulation, the user still able to navigate through our endpoints without getting blcoked.

Finally, `message.ack()` is used to mark that the message is received successfully.


Now, let's move to `main.py` and setup our FastAPI application.
```python3
from fastapi import FastAPI, Request, Response
import logging
from rmq import PikaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI()

@app.on_event("startup")
async def start_rmq():
    pass


@app.on_event("shutdown")
async def shutdown_rmq():
    pass


@app.get("/")
def root(response: Response):
    response.status_code = 200
    logger.info("hit root endpoint")
    return {"status_code": 200, "message": "Hello!"}


@app.get("/send-message")
async def send_message(request: Request, response: Response):
    pass
```
Here, we will have 2 endpoints for demonstrating our background jobs and regular endpoint for which user or another service can interact with our system. We also have both `startup` and `shutdown` FastAPI event which we will use it to start and stop our RMQ producer and consumer.

Let's start with setup the producer.
```python3
@app.on_event("startup")
async def start_rmq():
    # start producer
    app.rmq_producer = PikaClient(queue_name="test.queue",
                                  exchange_name="test.exchange",
                                  conn_str="amqp://root:root@127.0.0.1:5672")
    await app.rmq_producer.start_producer()


@app.on_event("shutdown")
async def shutdown_rmq():
    await app.rmq_producer.disconnect()
```
Quite simple, once the application start it will create a `PikaClient` object and call `start_producer()` to run it as producer. And once the application is stopped it will call the `disconnect()` to stop the producer.

Now, let's implement the similar thing to consumer with slight difference. We will run the consumer in a different thread so it won't block the thread that FastAPI used for serving endpoint routes.
```python3
def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    # inspired from https://gist.github.com/dmfigol/3e7d5b84a16d076df02baa9f53271058
    asyncio.set_event_loop(loop)
    loop.run_forever()


@app.on_event("startup")
async def start_rmq():
    # start producer
    app.rmq_producer = PikaClient(queue_name="test.queue",
                                        exchange_name="test.exchange",
                                        conn_str="amqp://root:root@127.0.0.1:5672")
    await app.rmq_producer.start_producer()

    # start consumer in other thread
    app.rmq_consumer = PikaClient(queue_name="test.queue",
                                  exchange_name="test.exchange",
                                  conn_str="amqp://root:root@127.0.0.1:5672")
    
    app.consumer_loop = asyncio.new_event_loop()
    tloop = threading.Thread(target=start_background_loop, args=(app.consumer_loop,), daemon=True)
    tloop.start()

    _ = asyncio.run_coroutine_threadsafe(app.rmq_consumer.start_consumer(), app.consumer_loop)


@app.on_event("shutdown")
async def shutdown_rmq():
    await app.rmq_producer.disconnect()
    await app.rmq_consumer.disconnect()

    app.consumer_loop.stop()
```
Here, inside the `startup` event we initiated a new asyncio event loop. Then, we trigger that loop to be run from another thread. We then delegate the `start_consumer()` calls to the loop, so the consumer will executed in the newly created thread. Finally, inside the `shutdown` event, we also call `disconnect()` to stop consumer RMQ from subscribing the queue. And we also trigger our created asyncio event loop to stop.

The final codes would be like this.

`rmq.py`
```python3
import logging
import aio_pika
import asyncio


logger = logging.getLogger(__name__)


class PikaClient():

    def __init__(self, queue_name: str, exchange_name: str, conn_str: str) -> None:
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.conn_str = conn_str

        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None


    async def start_connection(self):
        logger.info("Starting a new connection")
        self.connection = await aio_pika.connect_robust(url=self.conn_str)

        logger.info("Opening a new channel")
        self.channel = await self.connection.channel()

        logger.info("Declaring an exchange: %s" % self.exchange_name)
        self.exchange = await self.channel.declare_exchange(name=self.exchange_name, type=aio_pika.ExchangeType.DIRECT)

        await self.setup_queue()


    async def setup_queue(self):
        logger.info("Setup a queue: %s" % self.queue_name)
        self.queue = await self.channel.declare_queue(name=self.queue_name)

        logger.info("Bind queue to exchange")
        await self.queue.bind(self.exchange)


    async def start_producer(self):
        await self.start_connection()
        logger.info("Producer has been started")

        return self
        

    async def publish_message(self, message):
        await self.exchange.publish(
            aio_pika.Message(body=message.encode()),
            routing_key=self.queue_name
        )

    
    async def start_consumer(self):
        await self.start_connection()

        await self.channel.set_qos(prefetch_count=1)

        logger.info("Starting consumer")
        await self.queue.consume(self.handle_message)

        logger.info("Consumer has been started")

        return self
    

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage):

        # simulating long process
        await asyncio.sleep(10)

        logger.info("Consumer: Got message from producer: %s" % message.body.decode())

        await message.ack()


    async def disconnect(self):
        try:
            if not self.connection.is_closed:
                await self.connection.close()
        except Exception as _e:
            logger.error(_e)
```
{{< line_break >}}

`main.py`
```python3
from fastapi import FastAPI, Request, Response
import asyncio
import logging
from rmq import PikaClient
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI()


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    # inspired from https://gist.github.com/dmfigol/3e7d5b84a16d076df02baa9f53271058
    asyncio.set_event_loop(loop)
    loop.run_forever()


@app.on_event("startup")
async def start_rmq():
    # start producer
    app.rmq_producer = PikaClient(queue_name="test.queue",
                                        exchange_name="test.exchange",
                                        conn_str="amqp://root:root@127.0.0.1:5672")
    await app.rmq_producer.start_producer()

    # start consumer in other thread
    app.rmq_consumer = PikaClient(queue_name="test.queue",
                                  exchange_name="test.exchange",
                                  conn_str="amqp://root:root@127.0.0.1:5672")
    
    app.consumer_loop = asyncio.new_event_loop()
    tloop = threading.Thread(target=start_background_loop, args=(app.consumer_loop,), daemon=True)
    tloop.start()

    _ = asyncio.run_coroutine_threadsafe(app.rmq_consumer.start_consumer(), app.consumer_loop)


@app.on_event("shutdown")
async def shutdown_rmq():
    await app.rmq_producer.disconnect()
    await app.rmq_consumer.disconnect()

    app.consumer_loop.stop()


@app.get("/")
def root(response: Response):
    response.status_code = 200
    logger.info("hit root endpoint")
    return {"status_code": 200, "message": "Hello!"}


@app.get("/send-message")
async def send_message(request: Request, response: Response):
    message = "Hello from RMQ producer!"
    response.status_code = 202
    logger.info("message sent")
    await request.app.rmq_producer.publish_message(message)
    return {"status_code": 202, "message": "Your message has been sent."}
```

## Running the Application
When we start the server, shortly we will notice both producer and consumer has started.
![Starting server](images/serverstart.png#center)

Now, lets try to hit the long process endpoint following by several hits to the ordinary endpoint.
![Hit enpoint simulation](images/endpointhit.png#center)

Let's try several more then observe our RMQ dashboard.
![RMQ dashboard](images/rmq_dashboard.png#center)

And we can see there are some messages incoming and being processed.

## Conclusion
Now we know how to integrate both RMQ based publisher and subscriber within single FastAPI app.

If you have any inquiries, comments, suggestions, or critics please don’t hesitate to reach me out:

- Mail: affahrizain@gmail.com
- LinkedIn: https://www.linkedin.com/in/fahrizainn/
- GitHub: https://github.com/fhrzn

Until next time! 👋

---

## References
1. [https://www.rabbitmq.com/tutorials/tutorial-one-python.html](https://www.rabbitmq.com/tutorials/tutorial-one-python.html)
2. [https://aio-pika.readthedocs.io/en/latest/quick-start.html#asynchronous-message-processing](https://aio-pika.readthedocs.io/en/latest/quick-start.html#asynchronous-message-processing)
3. [RabbitMQ publisher and consumer with FastAPI by IT racer](https://itracer.medium.com/rabbitmq-publisher-and-consumer-with-fastapi-175fe87aefe1)
4. [Gist python asyncio event loop in a separated thread by dmfigol](https://gist.github.com/dmfigol/3e7d5b84a16d076df02baa9f53271058)