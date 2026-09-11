
# PSIQIT

[English](README.md) | **فارسی**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.3-green.svg)](https://github.com/psiqit/psiqit/releases)
[![CI](https://github.com/psiqit/psiqit/actions/workflows/ci.yml/badge.svg)](https://github.com/psiqit/psiqit/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1365410690.svg)](https://doi.org/10.5281/zenodo.22702331)

**ابزار اطلاعات کوانتومی علمی پایتون**

PSIQIT یک کتابخانه شبیه‌سازی محاسبات کوانتومی جامع و با کارایی بالا است که برای پژوهشگران، دانشجویان و توسعه‌دهندگان طراحی شده است. این کتابخانه چارچوبی قدرتمند برای شبیه‌سازی مدارهای کوانتومی، سیستم‌های کوانتومی باز، الگوریتم‌های تغییراتی و تصحیح خطای کوانتومی فراهم می‌کند.

## ویژگی‌های کلیدی

- **شبیه‌سازی مدار:** ساخت شهودی مدار با ترتیب کیوبیت LSB-first سازگار
- **الگوریتم‌های کوانتومی:** پیاده‌سازی‌های داخلی گروور، شور، دویچ-جوزا، برنشتاین-وازیرانی و QPE
- **سیستم‌های کوانتومی باز:** حل‌کننده‌های معادله مستر لیندبلاد، مسیرهای کوانتومی و تکامل زمانی
- **روش‌های تغییراتی:** چارچوب‌های VQE، QAOA و SSVQE با پشتیبانی از ansatz سفارشی
- **نظریه اطلاعات:** ابزارهایی برای محاسبه آنتروپی فون نویمن، همبستگی، منفی بودن و اطلاعات متقابل
- **تصحیح خطا:** پیاده‌سازی کدهای Bit-Flip، Phase-Flip، شور و استین

## نصب

نصب آخرین نسخه پایدار از طریق pip:

```bash
pip install psiqit
```

یا نصب مستقیم از مخزن برای آخرین ویژگی‌ها:

```bash
git clone https://github.com/psiqit/psiqit.git
cd psiqit
pip install -e .
```

## شروع سریع

### ایجاد حالت بل

```python
from psiqit.circuits import QuantumCircuit

# مقداردهی مدار ۲ کیوبیتی
circ = QuantumCircuit(2)

# اعمال هادامارد و CNOT
circ.h(0).cx(0, 1)

# اجرا و اندازه‌گیری
state = circ.run()
print("بردار حالت:", state.data)

result = circ.measure(shots=1024)
print("شمارش اندازه‌گیری:", result['counts'])
```

### اجرای جستجوی گروور

```python
from psiqit.algorithms import grover_search

# جستجو برای حالت |101> (5) در پایگاه داده ۳ کیوبیتی
result = grover_search(n_qubits=3, target=5, shots=1000)

print(f"هدف یافت شده: {result['most_likely']}")
print(f"احتمال موفقیت: {result['probability_target']:.2%}")
```

### حل معادله لیندبلاد

```python
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver

# تعریف همیلتونی و عملگرهای فروریزی
H = pauli_z()
L = [pauli_x()]

# حل معادله مستر
solver = LindbladSolver(H, L, gamma=[0.1])
result = solver.solve(t_span=(0, 10), dt=0.01)
```

## ساختار پروژه

```text
psiqit/
├── psiqit/                 # ماژول‌های اصلی کتابخانه
│   ├── quantum/            # حالت‌ها، عملگرها و اندازه‌گیری‌ها
│   ├── circuits/           # ساخت و اجرای مدار
│   ├── algorithms/         # الگوریتم‌های کوانتومی
│   ├── dynamics/           # لیندبلاد، تراتر و تکامل آدیاباتیک
│   ├── info/               # آنتروپی، وفاداری و درهم‌تنیدگی
│   ├── variational/        # چارچوب‌های VQE و QAOA
│   ├── error_correction/   # کدهای QEC
│   ├── qml/                # یادگیری ماشین کوانتومی
│   └── visualization/      # کره بلوخ، رسم مدار
├── tests/                  # مجموعه تست جامع
├── examples/               # نوت‌بوک‌های Jupyter و موارد استفاده
├── docs/                   # مستندات پروژه
└── pyproject.toml          # پیکربندی ساخت و بسته‌بندی
```

## مستندات

مستندات کامل در آدرس [https://psiqit.github.io/psiqit/](https://psiqit.github.io/psiqit/) در دسترس است.

مستندات شامل موارد زیر است:

- **مرجع API:** مستندات دقیق برای تمام ماژول‌ها و کلاس‌ها
- **آموزش‌ها:** راهنماهای گام‌به‌گام برای وظایف رایج محاسبات کوانتومی
- **مثال‌ها:** کاربردهای دنیای واقعی و موارد استفاده پژوهشی

## تست

PSIQIT با یک مجموعه تست دقیق ارائه می‌شود. برای اجرای تمام تست‌ها:

```bash
pytest tests/
# یا اجرای مجموعه جامع
python test_computational.py
```

## مشارکت

مشارکت‌ها مورد استقبال قرار می‌گیرند! لطفاً برای راهنما به [CONTRIBUTING.md](CONTRIBUTING.md) مراجعه کنید.

## مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات به فایل [LICENSE](LICENSE) مراجعه کنید.

## استناد

اگر از PSIQIT در پژوهش خود استفاده می‌کنید، لطفاً با استفاده از شناسه DOI زیر به آن استناد کنید. همچنین می‌توانید از دکمه‌ی **"Cite this repository"** در نوار کناری سمت راست همین صفحه برای دریافت فرمت‌های آماده استناد (مانند BibTeX، APA، IEEE و غیره) استفاده کنید.

[![DOI](https://zenodo.org/badge/1365410690.svg)](https://doi.org/10.5281/zenodo.22702331)

```bibtex
@software{psiqit2026,
  author       = {Azadmarzabadi, Mahdi},
  title        = {PSIQIT: Python Scientific Quantum Information Toolkit},
  version      = {1.0.3},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22702331},
  url          = {https://doi.org/10.5281/zenodo.22702331}
}
```

**فرمت APA:**
```text
Azadmarzabadi, M. (2026). PSIQIT: Python Scientific Quantum Information Toolkit (Version 1.0.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22702331
```

## تماس

**مهدی آزادمرزآبادی** - [psiqitofficial@protonmail.com](mailto:psiqitofficial@protonmail.com)  
لینک پروژه: [https://github.com/psiqit/psiqit](https://github.com/psiqit/psiqit)
