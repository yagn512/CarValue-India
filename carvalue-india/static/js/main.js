function readFields(container) {
  const data = {};
  container.querySelectorAll("input, select").forEach(el => { data[el.name] = el.value; });
  return data;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return res.json();
}

const predictForm = document.getElementById("predict-form");
if (predictForm) {
  predictForm.addEventListener("submit", async e => {
    e.preventDefault();
    const box = document.getElementById("predict-result");
    const err = document.getElementById("predict-error");
    box.hidden = true; err.hidden = true;
    try {
      const r = await postJSON("/api/predict", readFields(predictForm));
      if (!r.ok) { err.textContent = r.error; err.hidden = false; return; }
      document.getElementById("r-price").textContent = r.price_text;
      document.getElementById("r-range").textContent = r.low_text + " – " + r.high_text;
      document.getElementById("r-mae").textContent = r.mae_text;
      box.hidden = false;
    } catch (_) {
      err.textContent = "Could not reach the server. Please try again.";
      err.hidden = false;
    }
  });
}

const compareForm = document.getElementById("compare-form");
if (compareForm) {
  compareForm.addEventListener("submit", async e => {
    e.preventDefault();
    const box = document.getElementById("compare-result");
    const err = document.getElementById("compare-error");
    box.hidden = true; err.hidden = true;
    const body = {
      car_a: readFields(compareForm.querySelector('[data-car="car_a"]')),
      car_b: readFields(compareForm.querySelector('[data-car="car_b"]'))
    };
    try {
      const r = await postJSON("/api/compare", body);
      if (!r.ok) { err.textContent = r.error; err.hidden = false; return; }
      document.getElementById("a-price").textContent = r.car_a.price_text;
      document.getElementById("a-range").textContent = r.car_a.low_text + " – " + r.car_a.high_text;
      document.getElementById("b-price").textContent = r.car_b.price_text;
      document.getElementById("b-range").textContent = r.car_b.low_text + " – " + r.car_b.high_text;
      document.getElementById("c-verdict").textContent = r.verdict;
      box.hidden = false;
    } catch (_) {
      err.textContent = "Could not reach the server. Please try again.";
      err.hidden = false;
    }
  });
}
