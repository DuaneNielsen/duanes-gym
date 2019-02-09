class Event:
    def __init__(self, callback, *args, **kwargs):
        self.id = 0
        self.tick = 0
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        self.callback(*self.args, **self.kwargs)


class EventQueue:
    """Iterate over it every cycle to clear the events"""

    def __init__(self):
        self.queue = []
        self.ticker = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue and self.queue[0].tick <= self.ticker:
            return self.queue.pop(0)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.queue)

    def add(self, event, ticks=0):
        """
        Add an event to the queue
        :param event: can be a callable if no params are required, else wrap an Event object
        :param ticks: optional number of ticks to delay before firing the event
        """

        event = Event(event) if callable(event) else event
        event.tick = self.ticker + ticks

        if self.queue:
            for i, queued_event in enumerate(self.queue):
                if event.tick <= queued_event.tick:
                    self.queue.insert(i, event)
                    break
        else:
            self.queue.append(event)

    def tick(self):
        """ Event ticker call to advance the ticker"""
        self.ticker += 1

    def execute(self):
        for event in self:
            event.execute()