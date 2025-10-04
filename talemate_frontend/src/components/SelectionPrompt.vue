<template>
  <v-dialog v-model="open" max-width="600">
    <v-card>
      <v-card-title>
        <v-icon size="small" class="mr-2" color="primary">mdi-format-list-bulleted</v-icon>
        <span class="headline">{{ title }}</span>
      </v-card-title>
      <v-card-text>
        <div v-if="message" class="mb-3 prewrap text-body-2">{{ message }}</div>
        <v-radio-group v-model="selected" :disabled="sent">
          <v-radio
            v-for="(choice, idx) in choices"
            :key="idx"
            :label="choiceLabel(choice, idx)"
            :value="choiceValue(choice, idx)"
          ></v-radio>
        </v-radio-group>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn color="muted" variant="text" @click="cancel">Cancel</v-btn>
        <v-btn color="primary" variant="tonal" @click="confirm" :disabled="sent || !selected">Continue</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  name: 'SelectionPrompt',
  inject: ['getWebsocket', 'setWaitingForInput'],
  data() {
    return {
      open: false,
      title: 'Select an option',
      message: '',
      choices: [],
      selected: null,
      labels: [],
      sent: false,
    }
  },
  methods: {
    openPrompt(data) {
      this.sent = false;
      this.open = true;
      this.title = data.title || 'Select an option';
      this.message = data.message || '';
      this.choices = (data.data && data.data.choices) ? data.data.choices : [];
      this.labels = (data.data && data.data.labels) ? data.data.labels : [];
      const def = (data.data && data.data.default) ? data.data.default : (this.choices[0] || null);
      this.selected = def;
    },
    choiceLabel(choice, idx) {
      return this.labels[idx] || choice;
    },
    choiceValue(choice) {
      return choice;
    },
    confirm() {
      if (!this.selected) return;
      this.getWebsocket().send(JSON.stringify({ type: 'interact', text: this.selected }));
      this.setWaitingForInput(false);
      this.sent = true;
      this.open = false;
    },
    cancel() {
      const fallback = this.selected || (this.choices[0] || null);
      if (fallback) {
        this.getWebsocket().send(JSON.stringify({ type: 'interact', text: fallback }));
        this.setWaitingForInput(false);
      }
      this.sent = true;
      this.open = false;
    }
  }
}
</script>

<style scoped>
.prewrap { white-space: pre-wrap; }
</style>
