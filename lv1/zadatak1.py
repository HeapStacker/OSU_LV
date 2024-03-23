def total_euro(pay_per_hour, working_hours):
    return pay_per_hour * working_hours

radni_sati = int(input("Unesi br radnih sati: "))
placa_po_satu = int(input("Unesi placu po satu: "))
print("Radni sati: ", radni_sati)
print("eura/h: ", placa_po_satu)
print("Ukupno: ", total_euro(placa_po_satu, radni_sati), " eura")