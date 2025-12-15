(function() {
  const selectedParties = new Set();
  const selectedTypes = new Set();

  const partyPills = Array.from(document.querySelectorAll(".filter-pill[data-kind='party']"));
  const typePills = Array.from(document.querySelectorAll(".filter-pill[data-kind='type']"));
  const issueCards = Array.from(document.querySelectorAll(".issue-card"));

  function isVisible(card) {
    return card.style.display !== "none";
  }

  function recomputeCounts() {
    const partyCounts = {};
    const typeCounts = {};
    issueCards.forEach(card => {
      if (!isVisible(card)) return;
      let parties = [];
      try {
        parties = JSON.parse(card.dataset.parties || "[]");
      } catch (e) {
        parties = [];
      }
      const typ = card.dataset.type || "";
      parties.forEach(p => {
        partyCounts[p] = (partyCounts[p] || 0) + 1;
      });
      if (typ) {
        typeCounts[typ] = (typeCounts[typ] || 0) + 1;
      }
    });

    partyPills.forEach(pill => {
      const name = pill.dataset.value;
      const cnt = partyCounts[name] || 0;
      pill.textContent = `${name} (${cnt})`;
      pill.classList.toggle("disabled", cnt === 0);
    });
    typePills.forEach(pill => {
      const name = pill.dataset.value;
      const cnt = typeCounts[name] || 0;
      pill.textContent = `${name} (${cnt})`;
      pill.classList.toggle("disabled", cnt === 0);
    });
  }

  function applyFilters() {
    issueCards.forEach(card => {
      let parties = [];
      try {
        parties = JSON.parse(card.dataset.parties || "[]");
      } catch (e) {
        parties = [];
      }
      const typ = card.dataset.type || "";

      const partyMatch =
        selectedParties.size === 0 ||
        parties.some(p => selectedParties.has(p));
      const typeMatch =
        selectedTypes.size === 0 ||
        (typ && selectedTypes.has(typ));

      card.style.display = partyMatch && typeMatch ? "" : "none";
    });
    recomputeCounts();
  }

  function togglePill(pill) {
    const kind = pill.dataset.kind;
    const val = pill.dataset.value;
    const set = kind === "party" ? selectedParties : selectedTypes;
    if (set.has(val)) {
      set.delete(val);
      pill.classList.remove("active");
    } else {
      set.add(val);
      pill.classList.add("active");
    }
    applyFilters();
  }

  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", () => togglePill(pill));
  });

  // initial counts
  recomputeCounts();
})();
