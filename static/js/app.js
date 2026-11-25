/* =====================================================================
   RéussiteIA · interactions côté navigateur
   - page « Prédire » : curseurs synchronisés + prédiction en direct (API)
   - page « Lot » : glisser-déposer + téléchargement des résultats en CSV
   - page « Données » : filtre du tableau
   ===================================================================== */
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Page « Prédire »
  // ------------------------------------------------------------------
  const formulaire = document.getElementById("formulaire");
  if (formulaire) {
    const CIRCONFERENCE = 502.65;
    const curseurs = formulaire.querySelectorAll(".curseur");
    const resultat = document.getElementById("resultat");
    let minuterie = null;
    let numeroRequete = 0;

    const remplissage = (c) => {
      const pct = ((c.value - c.min) / (c.max - c.min)) * 100;
      c.style.setProperty("--pct", pct + "%");
    };

    const valeurs = () => {
      const v = {};
      formulaire.querySelectorAll('input[type="number"]').forEach((i) => { v[i.name] = i.value; });
      return v;
    };

    function afficher(r) {
      resultat.className = "carte resultat resultat--" + r.classe;
      document.getElementById("jauge-arc").setAttribute("stroke-dashoffset", (CIRCONFERENCE * (1 - r.probabilite / 100)).toFixed(2));
      const pourcent = document.getElementById("pourcent");
      pourcent.firstChild.textContent = Math.round(r.probabilite);
      document.getElementById("niveau").textContent = r.niveau;

      const phrase = document.getElementById("phrase");
      phrase.textContent = "Décision du modèle (seuil 50 %) : ";
      const fort = document.createElement("strong");
      fort.textContent = r.verdict;
      phrase.appendChild(fort);

      const av = document.getElementById("avertissements");
      av.replaceChildren(...r.avertissements.map((t) => {
        const d = document.createElement("div");
        d.className = "alerte alerte--info";
        d.textContent = t;
        return d;
      }));

      const liste = document.getElementById("liste-conseils");
      if (!r.conseils.length) {
        const li = document.createElement("li");
        li.className = "vide";
        li.textContent = "Aucun changement simple n'améliore nettement ce profil.";
        liste.replaceChildren(li);
      } else {
        liste.replaceChildren(...r.conseils.map((c) => {
          const li = document.createElement("li");
          const t = document.createElement("span");
          t.textContent = c.texte;
          const g = document.createElement("span");
          g.className = "gain";
          g.textContent = "+" + c.gain + " pts → " + c.apres + " %";
          li.append(t, g);
          return li;
        }));
      }
    }

    function afficherErreurs(erreurs) {
      const av = document.getElementById("avertissements");
      av.replaceChildren(...erreurs.map((t) => {
        const d = document.createElement("div");
        d.className = "alerte alerte--erreur";
        d.textContent = t;
        return d;
      }));
    }

    async function predire() {
      const n = ++numeroRequete;
      try {
        const rep = await fetch("/api/predire", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(valeurs()),
        });
        const donnees = await rep.json();
        if (n !== numeroRequete) return;          // une réponse plus récente est attendue
        if (rep.ok) afficher(donnees); else afficherErreurs(donnees.erreurs || ["Erreur inconnue."]);
      } catch (e) {
        afficherErreurs(["Impossible de contacter le serveur."]);
      }
    }

    const planifier = () => { clearTimeout(minuterie); minuterie = setTimeout(predire, 120); };

    curseurs.forEach((c) => {
      const champ = document.getElementById(c.dataset.cible);
      remplissage(c);
      c.addEventListener("input", () => { champ.value = c.value; remplissage(c); planifier(); });
      champ.addEventListener("input", () => {
        const v = parseFloat(String(champ.value).replace(",", "."));
        if (!Number.isNaN(v)) { c.value = v; remplissage(c); }
        planifier();
      });
    });

    // Profils d'exemple
    document.querySelectorAll("[data-valeurs]").forEach((b) => {
      b.addEventListener("click", () => {
        const vals = b.dataset.valeurs.split(",");
        curseurs.forEach((c, i) => {
          c.value = vals[i];
          document.getElementById(c.dataset.cible).value = vals[i];
          remplissage(c);
        });
        document.querySelectorAll("[data-valeurs]").forEach((x) => x.classList.toggle("puce--active", x === b));
        predire();
      });
    });

    // Avec JavaScript, pas besoin de recharger la page
    formulaire.addEventListener("submit", (e) => { e.preventDefault(); predire(); });
  }

  // ------------------------------------------------------------------
  // Page « Lot »
  // ------------------------------------------------------------------
  const fichier = document.getElementById("fichier");
  if (fichier) {
    const zone = document.getElementById("zone-depot");
    const nom = document.getElementById("nom-fichier");
    fichier.addEventListener("change", () => { if (fichier.files[0]) nom.textContent = fichier.files[0].name; });
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("survol"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("survol"); }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        fichier.files = e.dataTransfer.files;
        nom.textContent = e.dataTransfer.files[0].name;
      }
    });
  }

  const boutonCsv = document.getElementById("telecharger");
  const donneesLot = document.getElementById("donnees-lot");
  if (boutonCsv && donneesLot) {
    boutonCsv.addEventListener("click", () => {
      const lignes = JSON.parse(donneesLot.textContent);
      const cols = ["heures_etude", "heures_sommeil", "taux_presence", "note_precedente"];
      const echapper = (v) => '"' + String(v).replace(/"/g, '""') + '"';
      const csv = [["nom", ...cols, "probabilite_reussite", "prediction"].join(",")]
        .concat(lignes.map((l) => [echapper(l.id), ...cols.map((c) => l.valeurs[c]), l.probabilite, l.verdict].join(",")))
        .join("\n");
      const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "predictions.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }

  // ------------------------------------------------------------------
  // Page « Données » : filtre
  // ------------------------------------------------------------------
  const tableau = document.getElementById("tableau-donnees");
  if (tableau) {
    document.querySelectorAll("[data-filtre]").forEach((b) => {
      b.addEventListener("click", () => {
        const f = b.dataset.filtre;
        tableau.querySelectorAll("tbody tr").forEach((tr) => {
          tr.hidden = f !== "tous" && tr.dataset.reussite !== f;
        });
        document.querySelectorAll("[data-filtre]").forEach((x) => x.classList.toggle("puce--active", x === b));
      });
    });
  }
})();
