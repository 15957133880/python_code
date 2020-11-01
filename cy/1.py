class IngestionStage(object):

  def run(self):
    return 'Ingesting! Nom nom nom...'

class DigestionStage(object):

  def run(self, volume=1):
    return ' '.join(['Burp!'] * volume)

  def status(self):
    return 'Satiated.'

class Pipeline(object):

  def __init__(self):
    self.ingestion = IngestionStage()
    self.digestion = DigestionStage()

  def run(self):
    self.ingestion.run()
    self.digestion.run()

if __name__ == '__main__':
  fire.Fire(Pipeline)