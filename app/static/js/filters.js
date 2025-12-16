(function() {
  const selectedParties = new Set();
  const selectedTypes = new Set();
  const selectedAnswers = new Set();
  const WRITTEN_TYPE = "fyrirspurn til skrifl. svars";

  const partyPills = Array.from(document.querySelectorAll(".filter-pill[data-kind='party']"));
  const typePills = Array.from(document.querySelectorAll(".filter-pill[data-kind='type']"));
  const answerPills = Array.from(document.querySelectorAll(".filter-pill[data-kind='answer']"));
  const issueCards = Array.from(document.querySelectorAll(".issue-card"));

  function isVisible(card) {
    return card.style.display !== "none";
  }

  function recomputeCounts() {
    const partyCounts = {};
    const typeCounts = {};
    const answerCounts = {};
    issueCards.forEach(card => {
      if (!isVisible(card)) return;
      let parties = [];
      try {
        parties = JSON.parse(card.dataset.parties || "[]");
      } catch (e) {
        parties = [];
      }
      const typ = card.dataset.type || "";
      const ans = card.dataset.answer || "";
      parties.forEach(p => {
        partyCounts[p] = (partyCounts[p] || 0) + 1;
      });
      if (typ) {
        typeCounts[typ] = (typeCounts[typ] || 0) + 1;
      }
      if (ans && typ === WRITTEN_TYPE) {
        answerCounts[ans] = (answerCounts[ans] || 0) + 1;
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
    answerPills.forEach(pill => {
      const name = pill.dataset.value;
      const cnt = answerCounts[name] || 0;
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
      const ans = card.dataset.answer || "";

      // Answer filter only applies to written questions; hide others if an answer filter is active
      if (selectedAnswers.size > 0 && typ !== WRITTEN_TYPE) {
        card.style.display = "none";
        return;
      }

      const partyMatch =
        selectedParties.size === 0 ||
        parties.some(p => selectedParties.has(p));
      const typeMatch =
        selectedTypes.size === 0 ||
        (typ && selectedTypes.has(typ));
      const answerMatch =
        selectedAnswers.size === 0 ||
        (ans && selectedAnswers.has(ans));

      card.style.display = partyMatch && typeMatch && answerMatch ? "" : "none";
    });
    recomputeCounts();
  }

  function togglePill(pill) {
    const kind = pill.dataset.kind;
    const val = pill.dataset.value;
    const set = kind === "party" ? selectedParties : (kind === "type" ? selectedTypes : selectedAnswers);
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
