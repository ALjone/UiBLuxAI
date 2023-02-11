class Ingrid:
  def __init__(self) -> None:
    self.kos = 0

  def kos_med_ingrid(self):
    self.kos += 1

  @property
  def dobbel_kos(self):
    return self.kos**2



I = Ingrid()
for i in range(20):
  I.kos_med_ingrid()

print(I.dobbel_kos)