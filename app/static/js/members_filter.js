(() => {
  const select = document.getElementById("party-filter");
  if (!select) return;
  const panels = Array.from(document.querySelectorAll(".party-panel"));

  const applyFilter = () => {
    const val = select.value;
    panels.forEach(panel => {
      const match = !val || panel.dataset.party === val;
      panel.style.display = match ? "" : "none";
      if (match) {
        panel.open = true;
      }
    });
  };

  select.addEventListener("change", applyFilter);
  applyFilter();
})();
